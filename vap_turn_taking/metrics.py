import torch
from torchmetrics.classification.f_beta import F1Score
from torchmetrics import Metric, PrecisionRecallCurve, StatScores
from typing import Dict, List, Tuple, Optional

from vap_turn_taking.events import TurnTakingEvents


class F1_Hold_Shift(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_scores = StatScores(reduce="macro", multiclass=True, num_classes=2)

    def probs_shift_hold(self, p, shift, hold):
        probs, labels = [], []

        for next_speaker in [0, 1]:
            ws = torch.where(shift[..., next_speaker])
            if len(ws[0]) > 0:
                tmp_probs = p[ws][..., next_speaker]
                tmp_lab = torch.ones_like(tmp_probs, dtype=torch.long)
                probs.append(tmp_probs)
                labels.append(tmp_lab)

            # Hold label -> 0
            # Hold prob -> 1 - p  # opposite guess
            wh = torch.where(hold[..., next_speaker])
            if len(wh[0]) > 0:
                # complement in order to be combined with shifts
                tmp_probs = 1 - p[wh][..., next_speaker]
                tmp_lab = torch.zeros_like(tmp_probs, dtype=torch.long)
                probs.append(tmp_probs)
                labels.append(tmp_lab)

        if len(probs) > 0:
            probs = torch.cat(probs)
            labels = torch.cat(labels)
        else:
            probs = None
            labels = None
        return probs, labels

    def get_score(self, tp, fp, tn, fn, EPS=1e-9):
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return f1, precision, recall

    def reset(self):
        self.stat_scores.reset()

    def compute(self):
        hold, shift = self.stat_scores.compute()

        # HOLD
        h_tp, h_fp, h_tn, h_fn, h_sup = hold
        h_f1, h_precision, h_recall = self.get_score(h_tp, h_fp, h_tn, h_fn)

        # SHIFT
        s_tp, s_fp, s_tn, s_fn, s_sup = shift
        s_f1, s_precision, s_recall = self.get_score(s_tp, s_fp, s_tn, s_fn)

        # Weighted F1
        f1h = h_f1 * h_sup
        f1s = s_f1 * s_sup
        tot = h_sup + s_sup
        f1_weighted = (f1h + f1s) / tot
        return {
            "f1_weighted": f1_weighted,
            "hold": {
                "f1": h_f1,
                "precision": h_precision,
                "recall": h_recall,
                "support": h_sup,
            },
            "shift": {
                "f1": s_f1,
                "precision": s_precision,
                "recall": s_recall,
                "support": s_sup,
            },
        }

    def update(self, p, hold, shift):
        probs, labels = self.probs_shift_hold(p, shift=shift, hold=hold)
        if probs is not None:
            self.stat_scores.update(probs, labels)


class TurnTakingMetrics(Metric):
    """
    Used with discrete model, VAProjection.
    """

    def __init__(
        self,
        hs_kwargs,
        bc_kwargs,
        metric_kwargs,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.5,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
        frame_hz=100,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Metrics
        # self.f1: class to provide f1-weighted as well as other stats tp,fp,support, etc...
        self.hs = F1_Hold_Shift()
        self.predict_shift = F1Score(
            threshold=threshold_pred_shift,
            num_classes=2,
            multiclass=True,
            average="weighted",
        )
        self.short_long = F1Score(
            threshold=threshold_short_long,
            num_classes=2,
            multiclass=True,
            average="weighted",
        )
        self.predict_backchannel = F1Score(
            threshold=threshold_bc_pred,
            num_classes=2,
            multiclass=True,
            average="weighted",
        )

        self.pr_curve_bc_pred = bc_pred_pr_curve
        if self.pr_curve_bc_pred:
            self.bc_pred_pr = PrecisionRecallCurve(pos_label=1)

        self.pr_curve_shift_pred = shift_pred_pr_curve
        if self.pr_curve_shift_pred:
            self.shift_pred_pr = PrecisionRecallCurve(pos_label=1)

        self.pr_curve_long_short = long_short_pr_curve
        if self.pr_curve_long_short:
            self.long_short_pr = PrecisionRecallCurve(pos_label=1)

        # Extract the frames of interest for the given metrics
        self.eventer = TurnTakingEvents(
            hs_kwargs=hs_kwargs,
            bc_kwargs=bc_kwargs,
            metric_kwargs=metric_kwargs,
            frame_hz=frame_hz,
        )

    @torch.no_grad()
    def extract_events(self, va, max_frame=None):
        return self.eventer(va, max_frame=max_frame)

    def __repr__(self):
        s = "TurnTakingMetrics"
        s += self.eventer.__repr__()
        return s

    def update_short_long(self, p, short, long):
        """
        The given speaker in short/long is the one who initiated an onset.

        Use the backchannel (prediction) prob to recognize short utterance.

        event -> label
        short -> 1
        long -> 0
        """

        probs, labels = [], []

        # At the onset of a SHORT utterance the probability associated
        # with that person being the next speaker should be low -> 0
        if short.sum() > 0:
            w = torch.where(short)
            p_short = p[w]
            probs.append(p_short)
            # labels.append(torch.zeros_like(p_short))
            labels.append(torch.ones_like(p_short))

        # At the onset of a LONG utterance the probability associated
        # with that person being the next speaker should be high -> 1
        if long.sum() > 0:
            w = torch.where(long)
            p_long = p[w]
            probs.append(p_long)
            # labels.append(torch.ones_like(p_long))
            labels.append(torch.zeros_like(p_long))

        if len(probs) > 0:
            probs = torch.cat(probs)
            labels = torch.cat(labels).long()
            self.short_long.update(probs, labels)

            if self.pr_curve_long_short:
                self.long_short_pr.update(probs, labels)

    def update_predict_shift(self, p, pos, neg):
        """
        Predict upcomming speaker shift. The events pos/neg are given for the
        correct next speaker.

        correct classifications
        * pos next_speaker -> 1
        * neg next_speaker -> 1

        so we flip the negatives to have label 0 and take 1-p as their associated predictions

        """
        probs, labels = [], []

        # At the onset of a SHORT utterance the probability associated
        # with that person being the next speaker should be low -> 0
        if pos.sum() > 0:
            w = torch.where(pos)
            p_pos = p[w]
            probs.append(p_pos)
            labels.append(torch.ones_like(p_pos))

        # At the onset of a LONG utterance the probability associated
        # with that person being the next speaker should be high -> 1
        if neg.sum() > 0:
            w = torch.where(neg)
            p_neg = 1 - p[w]  # reverse to make negatives have label 0
            probs.append(p_neg)
            labels.append(torch.zeros_like(p_neg))

        if len(probs) > 0:
            probs = torch.cat(probs)
            labels = torch.cat(labels).long()
            self.predict_shift.update(probs, labels)

            if self.pr_curve_shift_pred:
                self.shift_pred_pr.update(probs, labels)

    def update_predict_backchannel(self, bc_pred_probs, pos, neg):
        """
        bc_pred_probs contains the probabilities associated with the given speaker
        initiating a backchannel in the "foreseeble" future.

        At POSITIVE events the speaker resposible for the actual upcomming backchannel
        is the same as the speaker in the event.

        At NEGATIVE events the speaker that "could have been" responsible for the upcomming backchennel
        is THE OTHER speaker so the probabilities much be switched.
        The probabilties associated with predicting THE OTHER is goin to say a backchannel is wrong so we
        flip the probabilities such that they should be close to 0.

        """
        probs, labels = [], []

        if pos.sum() > 0:
            w = torch.where(pos)
            p_pos = bc_pred_probs[w]
            probs.append(p_pos)
            labels.append(torch.ones_like(p_pos))

        if neg.sum() > 0:
            # where is negative samples?
            wb, wn, w_speaker = torch.where(neg)
            w_backchanneler = torch.logical_not(w_speaker).long()

            # p_neg = 1 - bc_pred_probs[(wb, wn, w_backchanneler)]
            p_neg = bc_pred_probs[(wb, wn, w_backchanneler)]
            probs.append(p_neg)
            labels.append(torch.zeros_like(p_neg))

        if len(probs) > 0:
            probs = torch.cat(probs)
            labels = torch.cat(labels).long()
            self.predict_backchannel(probs, labels)

            if self.pr_curve_bc_pred:
                self.bc_pred_pr.update(probs, labels)

    def reset(self):
        super().reset()
        self.hs.reset()
        self.predict_shift.reset()
        self.short_long.reset()
        self.predict_backchannel.reset()
        if self.pr_curve_bc_pred:
            self.bc_pred_pr.reset()

        if self.pr_curve_shift_pred:
            self.shift_pred_pr.reset()

        if self.pr_curve_long_short:
            self.long_short_pr.reset()

    def compute(self):
        f1_hs = self.hs.compute()
        f1_predict_shift = self.predict_shift.compute()
        f1_short_long = self.short_long.compute()

        ret = {
            "f1_hold_shift": f1_hs["f1_weighted"],
            "f1_predict_shift": f1_predict_shift,
            "f1_short_long": f1_short_long,
        }

        try:
            ret["f1_bc_prediction"] = self.predict_backchannel.compute()
        except:
            ret["f1_bc_prediction"] = -1

        if self.pr_curve_bc_pred:
            ret["pr_curve_bc_pred"] = self.bc_pred_pr.compute()

        if self.pr_curve_shift_pred:
            ret["pr_curve_shift_pred"] = self.shift_pred_pr.compute()

        if self.pr_curve_long_short:
            ret["pr_curve_long_short"] = self.long_short_pr.compute()

        ret["shift"] = f1_hs["shift"]
        ret["hold"] = f1_hs["hold"]
        return ret

    def update(self, p, bc_pred_probs=None, events=None, va=None):
        """
        p:              tensor, next_speaker probability. Must take into account current speaker such that it can be used for pre-shift/hold, backchannel-pred/ongoing
        pre_probs:      tensor, on active next speaker probability for independent
        bc_pred_probs:  tensor, Special probability associated with a backchannel prediction
        events:         dict, containing information about the events in the sequences
        vad:            tensor, VAD activity. Only used if events is not given.


        events: [
                    'shift',
                    'hold',
                    'short',
                    'long',
                    'predict_shift_pos',
                    'predict_shift_neg',
                    'predict_bc_pos',
                    'predict_bc_neg'
                ]
        """

        # Find valid event-frames if event is not given
        if events is None:
            events = self.extract_events(va)

        # SHIFT/HOLD
        self.hs.update(p, hold=events["hold"], shift=events["shift"])

        # Predict Shifts
        self.update_predict_shift(
            p, pos=events["predict_shift_pos"], neg=events["predict_shift_neg"]
        )

        # PREDICT BACKCHANNELS & Short/Long
        if bc_pred_probs is not None:
            self.update_predict_backchannel(
                bc_pred_probs,
                pos=events["predict_bc_pos"],
                neg=events["predict_bc_neg"],
            )

            # Long/Short
            self.update_short_long(
                bc_pred_probs, short=events["short"], long=events["long"]
            )
        else:
            # Long/Short
            self.update_short_long(p, short=events["short"], long=events["long"])


def main_old():
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    from conv_ssl.evaluation.utils import load_dm, load_model
    from conv_ssl.utils import to_device

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm(batch_size=12)
    # diter = iter(dm.val_dataloader())

    ###################################################
    # Load Model
    ###################################################
    # run_path = "how_so/VPModel/10krujrj"  # independent
    run_path = "how_so/VPModel/sbzhz86n"  # discrete
    # run_path = "how_so/VPModel/2608x2g0"  # independent (same bin size)
    model = load_model(run_path=run_path, strict=False)
    model = model.eval()
    # model = model.to("cpu")
    # model = model.to("cpu")

    event_kwargs = dict(
        shift_onset_cond=1,
        shift_offset_cond=1,
        hold_onset_cond=1,
        hold_offset_cond=1,
        min_silence=0.15,
        non_shift_horizon=2.0,
        non_shift_majority_ratio=0.95,
        metric_pad=0.05,
        metric_dur=0.1,
        metric_onset_dur=0.3,
        metric_pre_label_dur=0.5,
        metric_min_context=1.0,
        bc_max_duration=1.0,
        bc_pre_silence=1.0,
        bc_post_silence=3.0,
    )

    # # update vad_projection metrics
    # metric_kwargs = {
    #     "event_pre": 0.5,  # seconds used to estimate PRE-f1-SHIFT/HOLD
    #     "event_min_context": 1.0,  # min context duration before extracting metrics
    #     "event_min_duration": 0.15,  # the minimum required segment to extract SHIFT/HOLD (start_pad+target_duration)
    #     "event_horizon": 1.0,  # SHIFT/HOLD requires lookahead to determine mutual starts etc
    #     "event_start_pad": 0.05,  # Predict SHIFT/HOLD after this many seconds of silence after last speaker
    #     "event_target_duration": 0.10,  # duration of segment to extract each SHIFT/HOLD guess
    #     "event_bc_target_duration": 0.25,  # duration of activity, in a backchannel, to extract BC-ONGOING metrics
    #     "event_bc_pre_silence": 1,  # du
    #     "event_bc_post_silence": 2,
    #     "event_bc_max_active": 1.0,
    #     "event_bc_prediction_window": 0.4,
    #     "event_bc_neg_active": 1,
    #     "event_bc_neg_prefix": 1,
    #     "event_bc_ongoing_threshold": 0.5,
    #     "event_bc_pred_threshold": 0.5,
    # }
    # # Updatemetric_kwargs metrics
    # for metric, val in metric_kwargs.items():
    #     model.conf["vad_projection"][metric] = val

    N = 10
    model.test_metric = model.init_metric(
        model.conf, model.frame_hz, bc_pred_pr_curve=False, **event_kwargs
    )
    # tt_metrics = TurnTakingMetricsDiscrete(bin_times=model.conf['vad_projection']['bin_times'])
    for ii, batch in tqdm(enumerate(dm.val_dataloader()), total=N):
        batch = to_device(batch, model.device)
        ########################################################################
        # Extract events/labels on full length (with horizon) VAD
        events = model.test_metric.extract_events(batch["vad"], max_frame=1000)
        ########################################################################
        # Forward Pass through the model
        loss, out, batch = model.shared_step(batch)
        turn_taking_probs = model.get_next_speaker_probs(
            out["logits_vp"], vad=batch["vad"]
        )
        ########################################################################
        # Update metrics
        model.test_metric.update(
            p=turn_taking_probs["p"],
            pw=turn_taking_probs.get("pw", None),
            pre_probs=turn_taking_probs.get("pre_probs", None),
            bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
            events=events,
        )
        if ii == N:
            break
    result = model.test_metric.compute()
    print(result.keys())

    for k, v in result.items():
        print(f"{k}: {v}")


def how_to_use_with_model():
    import matplotlib.pyplot as plt
    from datasets_turntaking import DialogAudioDM
    from tqdm import tqdm

    from vap_turn_taking.plot_utils import plot_vad_oh
    from vap_turn_taking.events import TurnTakingEventsNew
    from vap_turn_taking.vap_new import VAP

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=20,
        batch_size=20,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup()

    objective = "comprative"
    eventer = TurnTakingEventsNew(max_time=dm.audio_duration)
    vap_objective = VAP(objective)
    print(dm)
    print(vap_objective)
    print("-" * 40)
    print(eventer)
    batch = next(iter(dm.val_dataloader()))
    print("batch['vad']: ", batch["vad"].shape)

    # Training
    # out = model(waveform=batch["waveform"], va=va_input, va_history=vah_input)
    ######################################################
    # Model forward
    ######################################################
    if objective == "independent":
        logits = torch.rand((*batch["vad"].shape[:-1], 2, vap_objective.n_bins))
    elif objective == "comparative":
        logits = torch.rand(batch["vad"].shape[:-1])
    else:
        logits = torch.rand((*batch["vad"].shape[:-1], vap_objective.n_classes))
    print("logits: ", tuple(logits.shape))

    ######################################################
    # VAP-head output: p, p_bc, labels, probs
    ######################################################
    vap_out = vap_objective(logits=logits, va=batch["vad"])
    for k, v in vap_out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    ######################################################
    # Events:   'pred_backchannel', 'pred_backchannel_neg',
    #           'shift', 'hold',
    #           'long', 'short',
    #           'pred_shift', 'pred_shift_neg'
    ######################################################
    events = eventer(batch["vad"])
    for k, v in events.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")

    # returns dict with keys: ['hs', 'pred_shift', 'ls', 'pred_backchannel']
    preds, targets = vap_objective.extract_prediction_and_targets(
        p=vap_out["p"], p_bc=vap_out["p_bc"], events=events
    )

    ###########################################
    # Initialize metrics
    ###########################################
    metric_shift_hold = F1Score(num_classes=2, average="weighted", multiclass=True)
    metric_pred_shift = F1Score(num_classes=2, average="weighted", multiclass=True)
    metric_pred_backchannel = F1Score(
        num_classes=2, average="weighted", multiclass=True
    )
    metric_long_short = F1Score(
        num_classes=2, threshold=0.5, average="weighted", multiclass=True
    )

    ###########################################
    # Update metrics (validation_step)
    ###########################################
    metric_shift_hold.update(preds=preds["hs"], target=targets["hs"])
    metric_pred_shift.update(preds=preds["pred_shift"], target=targets["pred_shift"])
    metric_pred_backchannel.update(
        preds=preds["pred_backchannel"], target=targets["pred_backchannel"]
    )
    metric_long_short.update(preds=preds["ls"], target=targets["ls"])
    sh = metric_shift_hold.compute()
    ps = metric_pred_shift.compute()
    pbc = metric_pred_backchannel.compute()
    ls = metric_long_short.compute()
    print("SH F1: ", sh)
    print("LS F1: ", ls)
    print("pred-S F1: ", ps)
    print("pred-BC F1: ", pbc)
    print("Shift vs Hold")
    print("fn: ", metric_shift_hold.fn)
    print("fp: ", metric_shift_hold.fp)
    print("tp: ", metric_shift_hold.tp)
    print("tn: ", metric_shift_hold.tn)
    print("support: ", metric_shift_hold.tp + metric_shift_hold.fn)

    b = 0
    fig, [ax, ax1] = plt.subplots(2, 1, figsize=(9, 6))
    _ = plot_vad_oh(batch["vad"][b], ax=ax)
    _ = plot_vad_oh(batch["vad"][b], ax=ax1)
    ax.axvline(eventer.min_context_frames, linewidth=4, color="k")
    ax.axvline(logits.shape[1], linewidth=4, color="k")
    for start, end, speaker in events["pred_shift_neg"][b]:
        ymin, ymax = 0, 1
        if speaker == 1:
            ymin, ymax = -1, 0
        ax1.vlines(start, ymin=ymin, ymax=ymax, color="g", linewidth=2)
        ax1.vlines(end, ymin=ymin, ymax=ymax, color="r", linewidth=2)
    plt.pause(0.1)


if __name__ == "__main__":
    from vap_turn_taking.config.example_data import example, event_conf

    metric = TurnTakingMetrics(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        bc_pred_pr_curve=True,
        shift_pred_pr_curve=True,
        long_short_pr_curve=True,
        frame_hz=100,
    )

    # Update
    metric.update(
        p=turn_taking_probs["p"],
        bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
        events=events,
    )

    # Compute
    result = metric.compute()
