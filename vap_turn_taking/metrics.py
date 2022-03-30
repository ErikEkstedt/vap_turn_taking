import torch
from torchmetrics import Metric, Accuracy, F1Score, PrecisionRecallCurve, StatScores

from vap_turn_taking.backchannel import (
    extract_backchannel_probs,
    extract_backchannel_prediction_probs,
)
from vap_turn_taking.events import TurnTakingEvents
from vap_turn_taking.vad import DialogEvents, VAD
from vap_turn_taking.vad_projection import VadLabel, ProjectionCodebook
from vap_turn_taking.utils import time_to_frames


@torch.no_grad()
def get_f1_statistics(ac, an, bc, bn):
    """
    F1 statistics over Shift/Hold

    Example 'Shift':
        * ac = A true positives
        * an = all A events, total
        * bc = B true positives
        * bn = all B events, total

    True Positives:  shift_correct
    False Negatives:  All HOLD predictions at SHIFT locations -> (shift_total - shift_correct)
    True Negatives:  All HOLD predictions at HOLD locations -> hold_correct
    False Positives:  All SHIFT predictions at HOLD locations -> (hold_total - hold_correct)

    Symmetrically true for Holds.
    """
    EPS = 1e-9
    tp = ac
    fn = an - ac
    tn = bc
    fp = bn - bc
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    support = tp + fn
    f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
    return {
        "f1": f1,
        "support": support,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


@torch.no_grad()
def correct_shift_hold(p, shift, hold, threshold=0.5):
    ret = {
        "shift": {"correct": 0, "n": 0},
        "hold": {"correct": 0, "n": 0},
    }
    # shifts
    next_speaker = 0
    w = torch.where(shift[..., next_speaker])
    if len(w[0]) > 0:
        sa = (p[w][..., next_speaker] >= threshold).sum().item()
        ret["shift"]["correct"] += sa
        ret["shift"]["n"] += len(w[0])
    next_speaker = 1
    w = torch.where(shift[..., next_speaker])
    if len(w[0]) > 0:
        sb = (p[w][..., next_speaker] >= threshold).sum().item()
        ret["shift"]["correct"] += sb
        ret["shift"]["n"] += len(w[0])
    # holds
    next_speaker = 0
    w = torch.where(hold[..., next_speaker])
    if len(w[0]) > 0:
        ha = (p[w][..., next_speaker] >= threshold).sum().item()
        ret["hold"]["correct"] += ha
        ret["hold"]["n"] += len(w[0])
    next_speaker = 1
    w = torch.where(hold[..., next_speaker])
    if len(w[0]) > 0:
        hb = (p[w][..., next_speaker] >= threshold).sum().item()
        ret["hold"]["correct"] += hb
        ret["hold"]["n"] += len(w[0])
    return ret


@torch.no_grad()
def extract_shift_hold_probs(p, shift, hold):
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


class F1_Hold_Shift(Metric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stat_scores = StatScores(reduce="macro", multiclass=True, num_classes=2)

    def get_score(self, tp, fp, tn, fn, EPS=1e-9):
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return f1, precision, recall

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
        probs, labels = extract_shift_hold_probs(p, shift=shift, hold=hold)
        if probs is not None:
            self.stat_scores.update(probs, labels)


class ShiftHoldMetric(Metric):
    """Used in conjuction with 'VadProjection' from datasets_turntaking"""

    def __init__(
        self,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        pre_active=0.5,
        bc_pre_silence=1.5,
        bc_post_silence=3,
        bc_max_active=2,
        frame_hz=100,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.pred_threshold = 0.5
        self.frame_hz = frame_hz

        # Shift/Holds
        self.frame_horizon = time_to_frames(horizon, frame_hz)
        self.frame_min_context = time_to_frames(min_context, frame_hz)
        self.frame_start_pad = time_to_frames(start_pad, frame_hz)
        self.frame_target_duration = time_to_frames(target_duration, frame_hz)
        self.frame_min_duration = self.frame_start_pad + self.frame_target_duration

        # pre-active hold/shift
        self.frame_pre_active = time_to_frames(pre_active, frame_hz)

        # backchannels
        self.bc_pre_silence_frames = time_to_frames(bc_pre_silence, frame_hz)
        self.bc_post_silence_frames = time_to_frames(bc_post_silence, frame_hz)
        self.bc_max_active_frames = time_to_frames(bc_max_active, frame_hz)

        # Hold/Shift (on_silence) data
        self.add_state("hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Pre - Hold/Shift data
        self.add_state(
            "pre_hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        # Backchannels
        self.add_state("bc_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bc_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def __repr__(self):
        s = "ShiftHoldMetric("
        s += f"\n  horizon: {self.frame_horizon}"
        s += f"\n  min_context: {self.frame_min_context}"
        s += f"\n  start_pad: {self.frame_start_pad}"
        s += f"\n  target_duration: {self.frame_target_duration}"
        s += f"\n  pre_active: {self.frame_pre_active}"
        s += f"\n  bc_pre_silence: {self.bc_pre_silence_frames}"
        s += f"\n  bc_post_silence: {self.bc_post_silence_frames}"
        s += f"\n  bc_max_active: {self.bc_max_active_frames}"
        s += f"\n  frame_hz: {self.frame_hz}"
        s += "\n)"
        return s

    def stats(self, ac, an, bc, bn):
        """
        F1 statistics over Shift/Hold

        Example 'Shift':
            * ac = shift_correct
            * an = shift_total
            * bc = hold_correct
            * bn = hold_total

        True Positives:  shift_correct
        False Negatives:  All HOLD predictions at SHIFT locations -> (shift_total - shift_correct)
        True Negatives:  All HOLD predictions at HOLD locations -> hold_correct
        False Positives:  All SHIFT predictions at HOLD locations -> (hold_total - hold_correct)

        Symmetrically true for Holds.
        """
        EPS = 1e-9
        tp = ac
        fn = an - ac
        tn = bc
        fp = bn - bc
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        support = tp + fn
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return {
            "f1": f1,
            "support": support,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def compute(self):
        """Compute final result"""
        stats = {
            "hold": self.stats(
                ac=self.hold_correct,
                an=self.hold_total,
                bc=self.shift_correct,
                bn=self.shift_total,
            ),
            "shift": self.stats(
                ac=self.shift_correct,
                an=self.shift_total,
                bc=self.hold_correct,
                bn=self.hold_total,
            ),
            "pre_hold": self.stats(
                ac=self.pre_hold_correct,
                an=self.pre_hold_total,
                bc=self.pre_shift_correct,
                bn=self.pre_shift_total,
            ),
            "pre_shift": self.stats(
                ac=self.pre_shift_correct,
                an=self.pre_shift_total,
                bc=self.pre_hold_correct,
                bn=self.pre_hold_total,
            ),
            "bc": self.bc_correct / self.bc_total,
        }

        # Weighted F1 score
        # scaled/weighted by the support of each metric
        # shift_f1*shift_support + hold_f1*hold_support )/ (shift_support + hold_support)
        f1h = stats["hold"]["f1"] * stats["hold"]["support"]
        f1s = stats["shift"]["f1"] * stats["shift"]["support"]
        tot = stats["hold"]["support"] + stats["shift"]["support"]
        stats["f1_weighted"] = (f1h + f1s) / tot

        f1h = stats["pre_hold"]["f1"] * stats["pre_hold"]["support"]
        f1s = stats["pre_shift"]["f1"] * stats["pre_shift"]["support"]
        tot = stats["pre_hold"]["support"] + stats["pre_shift"]["support"]
        stats["f1_pre_weighted"] = (f1h + f1s) / tot
        return stats

    def extract_acc(self, p_next, shift, hold):
        ret = {
            "shift": {"correct": 0, "n": 0},
            "hold": {"correct": 0, "n": 0},
        }
        # shifts
        next_speaker = 0
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sa = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sa
            ret["shift"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sb
            ret["shift"]["n"] += len(w[0])
        # holds
        next_speaker = 0
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            ha = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += ha
            ret["hold"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            hb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += hb
            ret["hold"]["n"] += len(w[0])
        return ret

    def extract_bc_acc(self, p_next, bc):
        ret = {"correct": 0, "n": 0}

        next_speaker = 0
        w = torch.where(bc[..., next_speaker])
        if len(w[0]) > 0:
            keep_turn = (
                (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            )
            ret["correct"] += keep_turn
            ret["n"] += len(w[0])

        next_speaker = 1
        w = torch.where(bc[..., next_speaker])
        if len(w[0]) > 0:
            keep_turn = (
                (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            )
            ret["correct"] += keep_turn
            ret["n"] += len(w[0])
        return ret

    def extract_events(self, vad, n_frames=None):
        hold, shift = DialogEvents.on_silence(
            vad,
            start_pad=self.frame_start_pad,
            target_frames=self.frame_target_duration,
            horizon=self.frame_horizon,
            min_context=self.frame_min_context,
            min_duration=self.frame_min_duration,
        )
        pre_hold, pre_shift = DialogEvents.get_active_pre_events(
            vad,
            hold,
            shift,
            start_pad=self.frame_start_pad,
            active_frames=self.frame_pre_active,
            min_context=self.frame_min_context,
        )
        backchannels = DialogEvents.extract_bc_candidates(
            vad,
            pre_silence_frames=self.bc_pre_silence_frames,
            post_silence_frames=self.bc_post_silence_frames,
            max_active_frames=self.bc_max_active_frames,
        )

        ret = {
            "hold": hold,
            "shift": shift,
            "pre_hold": pre_hold,
            "pre_shift": pre_shift,
            "backchannel": backchannels,
        }

        if n_frames is not None:
            for k, v in ret.items():
                ret[k] = v[:, :n_frames]

        return ret

    def event_update(self, p_next, events, bc_pre_probs=None):
        # extract TP, FP, TN, FN
        m = self.extract_acc(p_next, shift=events["shift"], hold=events["hold"])
        self.hold_correct += m["hold"]["correct"]
        self.hold_total += m["hold"]["n"]
        self.shift_correct += m["shift"]["correct"]
        self.shift_total += m["shift"]["n"]

        # Find active segment pre-events
        if bc_pre_probs is not None:
            m = self.extract_acc(
                bc_pre_probs, shift=events["pre_shift"], hold=events["pre_hold"]
            )
        else:
            m = self.extract_acc(
                p_next, shift=events["pre_shift"], hold=events["pre_hold"]
            )
        self.pre_hold_correct += m["hold"]["correct"]
        self.pre_hold_total += m["hold"]["n"]
        self.pre_shift_correct += m["shift"]["correct"]
        self.pre_shift_total += m["shift"]["n"]

        # Backchannels
        if bc_pre_probs is not None:
            m = self.extract_bc_acc(bc_pre_probs, events["backchannel"])
        else:
            m = self.extract_bc_acc(p_next, events["backchannel"])
        self.bc_correct += m["correct"]
        self.bc_total += m["n"]

    def update(self, p_next, vad, events=None, bc_pre_probs=None):
        # Find valid event-frames
        if events is None:
            events = self.extract_events(vad)
        self.event_update(p_next, events, bc_pre_probs=bc_pre_probs)


class TurnTakingMetricsOld(Metric):
    """
    Used with discrete model, VADProjection.
    """

    def __init__(
        self,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        pre_active=0.5,
        bc_pre_silence=1.5,
        bc_post_silence=3,
        bc_max_active=2,
        bc_prediction_window=0.5,
        bc_neg_active=1.0,
        bc_neg_prefix=1.0,
        frame_hz=100,
        bin_times=[0.2, 0.4, 0.6, 0.8],
        threshold=0.5,
        threshold_bc_ongoing=0.5,
        threshold_bc_pred=0.5,
        bc_pred_pr_curve=False,
        discrete=True,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Metrics
        # self.f1: class to provide f1-weighted as well as other stats tp,fp,support, etc...
        self.f1 = F1_Hold_Shift()
        self.f1_pre = F1Score(
            threshold=threshold, num_classes=2, multiclass=True, average="weighted"
        )
        self.bc_ongoing = F1Score(
            threshold=threshold_bc_ongoing,
            num_classes=2,
            multiclass=True,
            average="weighted",
        )
        self.bc_pred = F1Score(
            threshold=threshold_bc_pred,
            num_classes=2,
            multiclass=True,
            average="weighted",
        )
        # self.bc = Accuracy(threshold=threshold)
        # self.bc_pred = Accuracy(threshold=0.1)

        self.pr_curve_bc_pred = bc_pred_pr_curve
        if self.pr_curve_bc_pred:
            self.bc_pred_pr = PrecisionRecallCurve(pos_label=1)

        # Only available for discrete model
        self.discrete = discrete
        if self.discrete:
            self.f1_pw = F1Score(
                threshold=threshold, num_classes=2, multiclass=True, average="weighted"
            )

        # VadProjection Codebook
        self.codebook = ProjectionCodebook(bin_times=bin_times, frame_hz=frame_hz)
        self.labeler = VadLabel(bin_times, vad_hz=frame_hz)

        # Extract the frames of interest for the given metrics
        self.eventer = TurnTakingEventsOld(
            bc_idx=self.codebook.bc_prediction,
            horizon=horizon,
            min_context=min_context,
            start_pad=start_pad,
            target_duration=target_duration,
            bc_pre_silence=bc_pre_silence,
            bc_post_silence=bc_post_silence,
            bc_max_active=bc_max_active,
            bc_prediction_window=bc_prediction_window,
            pre_active=pre_active,  # shift/hold pre number of frames
            bc_neg_active=bc_neg_active,
            bc_neg_prefix=bc_neg_prefix,
            frame_hz=frame_hz,
        )

    @torch.no_grad()
    def extract_events(self, vad):
        projection_idx = self.codebook(self.labeler.vad_projection(vad))
        return self.eventer(vad, projection_idx)

    def __repr__(self):
        s = "TurnTakingMetrics"
        s += self.eventer.__repr__()
        return s

    @torch.no_grad()
    def compute(self):
        f1 = self.f1.compute()
        f1_pre = self.f1_pre.compute()
        f1_bc_ongoing = self.bc_ongoing.compute()
        # bc_ongoing_acc = self.bc.compute()

        ret = {
            "f1_weighted": f1["f1_weighted"],
            "f1_pre_weighted": f1_pre,
            "f1_bc_ongoing": f1_bc_ongoing,
        }

        try:
            ret["f1_bc_prediction"] = self.bc_pred.compute()
        except:
            pass

        if self.pr_curve_bc_pred:
            ret["pr_curve_bc_pred"] = self.bc_pred_pr.compute()

        ret["shift"] = f1["shift"]
        ret["hold"] = f1["hold"]

        # Extra metrics for discrete model
        if self.discrete:
            ret["f1_pw"] = self.f1_pw.compute()

        return ret

    def update_discrete(self, p, pw, events):
        """
        Metrics only defined for the 'discrete-vad-projection' models
        """
        # Pre-SHIFT/HOLD
        p_pre, label_pre = extract_shift_hold_probs(
            p, shift=events["pre_shift"], hold=events["pre_hold"]
        )
        if p_pre is not None:
            self.f1_pre.update(p_pre, label_pre)

        # PW-SHIFT/HOLD
        if pw is not None:
            probs, labels = extract_shift_hold_probs(
                pw, shift=events["shift"], hold=events["hold"]
            )
            if probs is not None:
                self.f1_pw.update(probs, labels)

    def update_independent(self, pre_probs, events):
        """
        Metrics are handled slightly differently 'independent-vad-projection' models
        """
        # Pre-SHIFT/HOLD
        p_pre, label_pre = extract_shift_hold_probs(
            pre_probs, shift=events["pre_shift"], hold=events["pre_hold"]
        )
        if p_pre is not None:
            self.f1_pre.update(p_pre, label_pre)

    @torch.no_grad()
    def update(
        self,
        p,
        pre_probs=None,
        pw=None,
        bc_pred_probs=None,
        events=None,
        vad=None,
    ):
        """
        p:      tensor, next_speaker probability. Must take into account current speaker such that it can be used for pre-shift/hold, backchannel-pred/ongoing
        pw:     tensor, probability ratio associated with each class weighted by the probability distribution at the current step  .
        bc_pred_probs:  tensor, Special probability associated with a backchannel prediction
        events:         dict, containing information about the events in the sequences
        vad:            tensor, VAD activity. Only used if events is not given.
        projection_idx: tensor, projection-indices labels of the sequence activity.
        """

        # Find valid event-frames if event is not given
        if events is None:
            events = self.extract_events(vad)

        # SHIFT/HOLD
        self.f1.update(p, hold=events["hold"], shift=events["shift"])

        # Backchannel
        bc_probs, bc_labels = extract_backchannel_probs(
            p, bc_pos=events["backchannel"], bc_neg=events["backchannel_neg"]
        )
        if bc_probs is not None:
            self.bc_ongoing.update(bc_probs, bc_labels)

        # Backchannel Prediction
        if bc_pred_probs is not None:
            bc_pred, bc_pred_lab = extract_backchannel_prediction_probs(
                bc_pred_probs,
                bc_pred_pos=events["backchannel_prediction"],
                bc_pred_neg=events["backchannel_prediction_neg"],
            )

            if bc_pred is not None:
                self.bc_pred.update(bc_pred, bc_pred_lab)
                if self.pr_curve_bc_pred:
                    self.bc_pred_pr.update(bc_pred, bc_pred_lab)

        # Some metrics differ dependent on model
        if self.discrete:
            self.update_discrete(p, pw, events)
        else:
            self.update_independent(pre_probs, events)


class TurnTakingMetrics(Metric):
    """
    Used with discrete model, VADProjection.
    """

    def __init__(
        self,
        threshold_pred_shift=0.5,
        threshold_short_long=0.5,
        threshold_bc_pred=0.5,
        bc_pred_pr_curve=False,
        shift_pred_pr_curve=False,
        long_short_pr_curve=False,
        frame_hz=100,
        dist_sync_on_step=False,
        **event_kwargs,
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
        self.eventer = TurnTakingEvents(**event_kwargs, frame_hz=frame_hz)

    @torch.no_grad()
    def extract_events(self, vad, max_frame=1000):
        return self.eventer(vad, max_frame=max_frame)

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

    def update(
        self, p, pre_probs=None, bc_pred_probs=None, events=None, vad=None, **kwargs
    ):
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
            events = self.extract_events(vad)

        # SHIFT/HOLD
        self.hs.update(p, hold=events["hold"], shift=events["shift"])

        # PREDICT BACKCHANNELS
        if bc_pred_probs is not None:
            self.update_predict_backchannel(
                bc_pred_probs,
                pos=events["predict_bc_pos"],
                neg=events["predict_bc_neg"],
            )

        # Long/Short
        if pre_probs is None:
            self.update_predict_shift(
                p, pos=events["predict_shift_pos"], neg=events["predict_shift_neg"]
            )
            self.update_short_long(
                bc_pred_probs, short=events["short"], long=events["long"]
            )
        else:
            self.update_predict_shift(
                p,
                pos=events["predict_shift_pos"],
                neg=events["predict_shift_neg"],
            )
            self.update_short_long(
                pre_probs, short=events["short"], long=events["long"]
            )


if __name__ == "__main__":
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

    # print("f1_weighted: ", result["f1_weighted"])
    # print("f1_pre_weighted: ", result["f1_pre_weighted"])
    # print("f1_bc_ongoing: ", result["f1_bc_ongoing"])
    # print("f1_bc_prediction: ", result["f1_bc_prediction"])

    # precision, recall, threshold = result["pr_curve_bc_pred"]
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(recall.cpu(), precision.cpu())
    # ax.set_xlabel("Recall")
    # ax.set_xlim([0, 1])
    # ax.set_ylabel("Precision")
    # ax.set_ylim([0, 1])
    # plt.show()
