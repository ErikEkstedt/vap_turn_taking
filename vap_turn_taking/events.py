import torch
import numpy.random as np_random
import random

from vap_turn_taking.backchannel import Backchannel, BackchannelNew
import vap_turn_taking.functional as VF
from vap_turn_taking.hold_shifts import HoldShift, HoldShiftNew
from vap_turn_taking.utils import (
    time_to_frames,
    find_island_idx_len,
    get_dialog_states,
    get_last_speaker,
)


class TurnTakingEvents:
    def __init__(
        self,
        hs_kwargs,
        bc_kwargs,
        metric_kwargs,
        frame_hz=50,
    ):
        self.frame_hz = frame_hz

        # Times to frames
        self.metric_kwargs = self.kwargs_to_frames(metric_kwargs, frame_hz)
        self.hs_kwargs = self.kwargs_to_frames(hs_kwargs, frame_hz)
        self.bc_kwargs = self.kwargs_to_frames(bc_kwargs, frame_hz)

        # values for metrics
        self.metric_min_context = self.metric_kwargs["min_context"]
        self.metric_pad = self.metric_kwargs["pad"]
        self.metric_dur = self.metric_kwargs["dur"]
        self.metric_onset_dur = self.metric_kwargs["onset_dur"]
        self.metric_pre_label_dur = self.metric_kwargs["pre_label_dur"]

        self.HS = HoldShift(**self.hs_kwargs)
        self.BS = Backchannel(**self.bc_kwargs)

    def kwargs_to_frames(self, kwargs, frame_hz):
        new_kwargs = {}
        for k, v in kwargs.items():
            new_kwargs[k] = time_to_frames(v, frame_hz)
        return new_kwargs

    def __repr__(self):
        s = "TurnTakingEvents\n"
        s += str(self.HS) + "\n"
        s += str(self.BS)
        return s

    def count_occurances(self, x):
        n = 0
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                _, _, v = find_island_idx_len(x[b, :, sp])
                n += (v == 1).sum().item()
        return n

    def sample_negative_segments(self, x, n):
        """
        Used to pick a subset of negative segments.
        That is on events where the negatives are constrained in certain
        single chunk segments.

        Used to sample negatives for LONG/SHORT prediction.

        all start onsets result in either longer or shorter utterances.
        Utterances defined as backchannels are considered short and utterances
        after pauses or at shifts are considered long.

        """
        neg_candidates = []
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                starts, durs, v = find_island_idx_len(x[b, :, sp])

                starts = starts[v == 1]
                durs = durs[v == 1]
                for s, d in zip(starts, durs):
                    neg_candidates.append([b, s, s + d, sp])

        sampled_negs = torch.arange(len(neg_candidates))
        if len(neg_candidates) > n:
            sampled_negs = np_random.choice(sampled_negs, size=n, replace=False)

        negs = torch.zeros_like(x)
        for ni in sampled_negs:
            b, s, e, sp = neg_candidates[ni]
            negs[b, s:e, sp] = 1.0

        return negs.float()

    def sample_negatives(self, x, n, dur):
        """

        Choose negative segments from x which contains long stretches of
        possible starts of the negative segments.

        Used to sample negatives from NON-SHIFTS which represent longer segments
        where every frame is a possible prediction point.

        """

        onset_pad_min = 3
        onset_pad_max = 10

        neg_candidates = []
        for b in range(x.shape[0]):
            for sp in [0, 1]:
                starts, durs, v = find_island_idx_len(x[b, :, sp])

                starts = starts[v == 1]
                durs = durs[v == 1]

                # Min context condition
                durs = durs[starts >= self.metric_min_context]
                starts = starts[starts >= self.metric_min_context]

                # Over minimum duration condition
                starts = starts[durs > dur]
                durs = durs[durs > dur]

                if len(starts) == 0:
                    continue

                for s, d in zip(starts, durs):
                    # end of valid frames minus duration of concurrent segment
                    end = s + d - dur

                    if end - s <= onset_pad_min:
                        onset_pad = 0
                    elif end - s <= onset_pad_max:
                        onset_pad = onset_pad_min
                    else:
                        onset_pad = torch.randint(onset_pad_min, onset_pad_max, (1,))[
                            0
                        ].item()

                    for neg_start in torch.arange(s + onset_pad, end, dur):
                        neg_candidates.append([b, neg_start, sp])

        sampled_negs = torch.arange(len(neg_candidates))
        if len(neg_candidates) > n:
            sampled_negs = np_random.choice(sampled_negs, size=n, replace=False)

        negs = torch.zeros_like(x)
        for ni in sampled_negs:
            b, s, sp = neg_candidates[ni]
            negs[b, s : s + dur, sp] = 1.0
        return negs.float()

    def __call__(self, vad, max_frame=None):
        ds = get_dialog_states(vad)
        last_speaker = get_last_speaker(vad, ds)

        # TODO:
        # TODO: having all events as a list/dict with (b, start, end, speaker) may be very much faster?
        # TODO:

        # HOLDS/SHIFTS:
        # shift, pre_shift, long_shift_onset,
        # hold, pre_hold, long_hold_onset,
        # shift_overlap, pre_shift_overlap, non_shift
        tt = self.HS(
            vad=vad, ds=ds, max_frame=max_frame, min_context=self.metric_min_context
        )

        # Backchannels: backchannel, pre_backchannel
        bcs = self.BS(
            vad=vad,
            last_speaker=last_speaker,
            max_frame=max_frame,
            min_context=self.metric_min_context,
        )

        #######################################################
        # LONG/SHORT
        #######################################################
        # Investigate the model output at the start of an IPU
        # where SHORT segments are "backchannel" and LONG har onset on new TURN (SHIFTs)
        # or onset of HOLD ipus
        short = bcs["backchannel"]
        long = self.sample_negative_segments(tt["long_shift_onset"], 1000)

        #######################################################
        # Predict shift
        #######################################################
        # Pos: window, on activity, prior to EOT before a SHIFT
        # Neg: Sampled from NON-SHIFT, on activity.
        n_predict_shift = self.count_occurances(tt["pre_shift"])
        if n_predict_shift == 0:
            predict_shift_neg = torch.zeros_like(tt["pre_shift"])
        else:
            # NON-SHIFT where someone is active
            activity = ds == 0  # only A
            activity = torch.logical_or(activity, ds == 3)  # AND only B
            activity = activity[:, : tt["non_shift"].shape[1]].unsqueeze(-1)
            non_shift_on_activity = torch.logical_and(tt["non_shift"], activity)
            predict_shift_neg = self.sample_negatives(
                non_shift_on_activity, n_predict_shift, dur=self.metric_pre_label_dur
            )

        #######################################################
        # Predict backchannels
        #######################################################
        # Pos: 0.5 second prior a backchannel
        # Neg: Sampled from NON-SHIFT, everywhere
        n_pre_bc = self.count_occurances(bcs["pre_backchannel"])
        if n_pre_bc == 0:
            predict_bc_neg = torch.zeros_like(bcs["pre_backchannel"])
        else:
            predict_bc_neg = self.sample_negatives(
                tt["non_shift"], n_pre_bc, dur=self.metric_pre_label_dur
            )

        # return tt
        return {
            "shift": tt["shift"][:, :max_frame],
            "hold": tt["hold"][:, :max_frame],
            "short": short[:, :max_frame],
            "long": long[:, :max_frame],
            "predict_shift_pos": tt["pre_shift"][:, :max_frame],
            "predict_shift_neg": predict_shift_neg[:, :max_frame],
            "predict_bc_pos": bcs["pre_backchannel"][:, :max_frame],
            "predict_bc_neg": predict_bc_neg[:, :max_frame],
        }


class TurnTakingEventsNew:
    def __init__(
        self,
        sh_pre_cond_time: float = 1.0,
        sh_post_cond_time: float = 1.0,
        sh_prediction_region_on_active: bool = True,
        bc_pre_cond_time: float = 1.0,
        bc_post_cond_time: float = 1.0,
        bc_max_duration: float = 1.0,
        bc_negative_pad_left_time: float = 1.0,
        bc_negative_pad_right_time: float = 2.0,
        prediction_region_time: float = 0.5,
        long_onset_region_time: float = 0.2,
        long_onset_condition_time: float = 1.0,
        min_context_time: float = 3,
        metric_time: float = 0.2,
        metric_pad_time: float = 0.05,
        max_time: int = 10,
        frame_hz: int = 50,
        equal_hold_shift: bool = True,
    ):
        self.frame_hz = frame_hz
        self.equal_hold_shift = equal_hold_shift
        self.min_silence_time = metric_time
        self.metric_time = metric_time
        self.metric_pad_time = metric_pad_time
        self.min_silence_time = metric_pad_time + metric_time

        # Global
        self.prediction_region_frames = time_to_frames(prediction_region_time, frame_hz)
        self.min_context_frames = time_to_frames(min_context_time, frame_hz)
        self.metric_frames = time_to_frames(metric_time, frame_hz)
        self.metric_pad_frames = time_to_frames(metric_pad_time, frame_hz)
        self.max_frames = time_to_frames(max_time, frame_hz)

        # Shift/Hold
        self.sh_pre_cond_frames = time_to_frames(sh_pre_cond_time, frame_hz)
        self.sh_post_cond_frames = time_to_frames(sh_post_cond_time, frame_hz)
        self.sh_prediction_region_on_active = sh_prediction_region_on_active

        # Backchannel
        self.bc_pre_cond_frames = time_to_frames(bc_pre_cond_time, frame_hz)
        self.bc_post_cond_frames = time_to_frames(bc_post_cond_time, frame_hz)
        self.bc_negative_pad_left_frames = time_to_frames(
            bc_negative_pad_left_time, frame_hz
        )
        self.bc_negative_pad_right_frames = time_to_frames(
            bc_negative_pad_right_time, frame_hz
        )
        self.bc_max_duration = time_to_frames(bc_max_duration, frame_hz)

        # Long/Short
        self.long_onset_region_frames = time_to_frames(long_onset_region_time, frame_hz)
        self.long_onset_condition_frames = time_to_frames(
            long_onset_condition_time, frame_hz
        )

        # Memory to add extra event in upcomming batches
        # if there is a discrepancy between
        # `pred_shift` & `pred_shift_neg` and
        # `pred_bc` & `pred_bc_neg` and
        self.add_extra = {"shift": 0, "pred_shift": 0, "pred_backchannel": 0}

        assert (
            min_context_time < max_time
        ), "`minimum_context_time` must be lower than `max_time`"

        self.HS = HoldShiftNew(
            pre_cond_time=sh_pre_cond_time,
            post_cond_time=sh_post_cond_time,
            prediction_region_time=prediction_region_time,
            prediction_region_on_active=sh_prediction_region_on_active,
            long_onset_condition_time=long_onset_condition_time,
            long_onset_region_time=long_onset_region_time,
            min_silence_time=self.min_silence_time,
            min_context_time=min_context_time,
            max_time=max_time,
            frame_hz=frame_hz,
        )

        self.BC = BackchannelNew(
            pre_cond_time=bc_pre_cond_time,
            post_cond_time=bc_post_cond_time,
            prediction_region_time=prediction_region_time,
            negative_pad_left_time=bc_negative_pad_left_time,
            negative_pad_right_time=bc_negative_pad_right_time,
            max_bc_duration=bc_max_duration,
            min_context_time=min_context_time,
            max_time=max_time,
            frame_hz=frame_hz,
        )

    def __repr__(self) -> str:
        s = "TurnTakingEvents\n\n"
        s += self.BC.__repr__() + "\n"
        s += self.HS.__repr__()
        return s

    def sample_equal_amounts(self, a_set, b_set, event_type, is_backchannel=False):
        """Sample a subset from `b_set` of equal size of `a_set`"""

        batch_size = len(a_set)
        n_to_sample = sum([len(events) for events in a_set])

        # Create empty set
        subset = [[] for _ in range(batch_size)]

        # Flatten all events in B
        b_set_flat, batch_idx = [], []
        for b in range(batch_size):
            b_set_flat += b_set[b]
            batch_idx += [b] * len(b_set[b])

        # The maximum number of samples to sample
        n_max = len(b_set_flat)

        if n_max < n_to_sample:
            diff = n_to_sample - n_max
            self.add_extra[event_type] += diff
            n_to_sample = n_max
        else:
            diff = n_max - n_to_sample
            add_extra = min(diff, self.add_extra[event_type])
            n_to_sample += add_extra  # add extra 'negatives'
            # subtract the number of extra events we now sample
            self.add_extra[event_type] -= add_extra

        # Choose random a random subset from b_set
        for idx in random.sample(list(range(len(b_set_flat))), k=n_to_sample):
            b = batch_idx[idx]
            entry = b_set_flat[idx]
            if is_backchannel:
                entry = self.BC.sample_negative_segment(entry)
            subset[b].append(entry)
        return subset

    @torch.no_grad()
    def __call__(self, vad: torch.Tensor):
        assert (
            vad.ndim == 3
        ), f"Expects vad of shape (B, N_FRAMES, 2) but got {vad.shape}"
        ret = {}

        ds = VF.get_dialog_states(vad)
        bc = self.BC(vad, ds=ds)
        hs = self.HS(vad, ds=ds)

        ret.update(bc)
        ret.update(hs)

        # Sample equal amounts of "pre-hold" regions as "pre-shift"
        # ret["pred_shift_neg"] = self.sample_pred_shift_negatives(ret)
        ret["pred_shift_neg"] = self.sample_equal_amounts(
            ret["pred_shift"], ret["pred_hold"], event_type="pred_shift"
        )

        # Sample equal amounts of "pred_backchannel_neg" as "pred_backchannel"
        # `if_backchannel=True`:
        #    from the found 'regions of possible negative backchannel prediction segments'
        #    sample the actual region to be used
        ret["pred_backchannel_neg"] = self.sample_equal_amounts(
            ret["pred_backchannel"],
            ret["pred_backchannel_neg"],
            event_type="pred_backchannel",
            is_backchannel=True,
        )

        if self.equal_hold_shift:
            ret["hold"] = self.sample_equal_amounts(
                ret["shift"], ret["hold"], event_type="shift"
            )
        # renames
        ret["short"] = ret.pop("backchannel")
        return ret


def _old_main():
    import matplotlib.pyplot as plt
    from vap_turn_taking.config.example_data import example, event_conf
    from vap_turn_taking.plot_utils import plot_vad_oh, plot_event

    eventer = TurnTakingEvents(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        frame_hz=50,
    )
    va = example["va"]
    events = eventer(va, max_frame=None)
    print("long: ", (events["long"] != example["long"]).sum())
    print("short: ", (events["short"] != example["short"]).sum())
    print("shift: ", (events["shift"] != example["shift"]).sum())
    print("hold: ", (events["hold"] != example["hold"]).sum())
    # for k, v in events.items():
    #     if isinstance(v, torch.Tensor):
    #         print(f"{k}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k}: {v}")
    fig, ax = plot_vad_oh(va[0])
    # _, ax = plot_event(events["shift"][0], ax=ax)
    _, ax = plot_event(events["hold"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(events["short"][0], ax=ax)
    # _, ax = plot_event(events["long"][0], color=['r', 'r'], ax=ax)
    # _, ax = plot_event(example['short'][0], color=["g", "g"], ax=ax)
    # _, ax = plot_event(example['long'][0], color=["r", "r"], ax=ax)
    _, ax = plot_event(example["hold"][0], color=["b", "b"], ax=ax)
    # _, ax = plot_event(example['shift'][0], color=["g", "g"], ax=ax)
    # _, ax = plot_event(example['short'][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(example['long'][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(bc[0], color=["b", "b"], ax=ax)
    # _, ax = plot_event(tt["shift_overlap"][0], ax=ax)
    # _, ax = plot_event(events["short"][0], color=["b", "b"], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt_bc["pre_backchannel"][0], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt["hold"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(tt['pre_shift'][0], color=['g', 'g'], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt['pre_hold'][0], color=['r', 'r'], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt['long_shift_onset'][0], color=['r', 'r'], alpha=0.2, ax=ax)
    # _, ax = plot_event(events["non_shift"][0], color=["r", "r"], alpha=0.2, ax=ax)
    plt.pause(0.1)


def _time_comparison():
    import timeit
    from vap_turn_taking.config.example_data import event_conf

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    vad = torch.cat([vad] * 10)
    eventerOld = TurnTakingEvents(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        frame_hz=50,
    )
    eventer = TurnTakingEventsNew()

    out = eventer(vad)

    old = timeit.timeit("eventerOld(vad)", globals=globals(), number=50)
    new = timeit.timeit("eventer(vad)", globals=globals(), number=50)
    print(f"OLD {round(old, 3)}s vs {round(new,3)}s NEW")
    if old > new:
        print(
            f"NEW approach is {round(old/new,3)} times or {round(100*old/new - 100 ,1)}% faster!"
        )
    else:
        print(f"OLD approach is {round(new/old,3)} times faster!")
        print(f"OLD approach is {round(100*new/old - 100 ,1)}% faster!")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from vap_turn_taking.plot_utils import plot_vad_oh
    from datasets_turntaking import DialogAudioDM
    from tqdm import tqdm

    dm = DialogAudioDM(
        datasets=["switchboard", "fisher"],
        audio_duration=20,
        batch_size=8,
        num_workers=4,
    )
    dm.prepare_data()
    dm.setup()

    eventer = TurnTakingEventsNew()
    print(eventer)
    vad = torch.load("example/vap_data.pt")["bc"]["vad"]
    events = eventer(vad)
    events

    # batch = next(iter(dm.val_dataloader()))
    # ret = eventer(batch["vad"])

    n_events = {
        "shift": 0,
        "hold": 0,
        "pred_shift": 0,
        "pred_shift_neg": 0,
        "pred_backchannel": 0,
        "pred_backchannel_neg": 0,
        "long": 0,
        "short": 0,
    }
    for batch in tqdm(dm.val_dataloader()):
        batch_size = batch["vad"].shape[0]
        events = eventer(batch["vad"])
        for b in range(batch_size):
            n_events["shift"] += len(events["shift"][b])
            n_events["hold"] += len(events["hold"][b])
            n_events["long"] += len(events["long"][b])
            n_events["short"] += len(events["short"][b])
            n_events["pred_shift"] += len(events["pred_shift"][b])
            n_events["pred_shift_neg"] += len(events["pred_shift_neg"][b])
            n_events["pred_backchannel"] += len(events["pred_backchannel"][b])
            n_events["pred_backchannel_neg"] += len(events["pred_backchannel_neg"][b])
            # n_events["pred_bc_neg"] += len(events["pred_backchannel_neg"][b])
    for k, v in n_events.items():
        print(f"{k}: {v}")
    # print("Add extra pred shift neg: ", eventer.add_extra_pred_shift_neg)
    # print("Add extra bc shift neg: ", eventer.add_extra_pred_bc_neg)
    for k, v in eventer.add_extra.items():
        print(f"Add extra '{k}': {v}")

    # for b in range(vad.shape[0]):
    #     fig, [ax, ax1] = plt.subplots(2, 1, figsize=(9, 6))
    #     _ = plot_vad_oh(vad[b], ax=ax)
    #     # _ = plot_vad_oh(filled_vad, ax=ax1)
    #     ax.axvline(eventer.min_context_frames, linewidth=4, color="k")
    #     ax.axvline(eventer.max_frames, linewidth=4, color="k")
    #     for start, end, speaker in events["pred_shift"][b]:
    #         ax.axvline(start, linewidth=4, color="g")
    #         ax.axvline(end, linewidth=4, color="g")
    #     for start, end, speaker in events["pred_hold"][b]:
    #         ax.axvline(start, linewidth=4, linestyle="dashed", color="r")
    #         ax.axvline(end, linewidth=4, linestyle="dashed", color="r")
    #     plt.show()
