import torch

from vad_turn_taking.utils import time_to_frames
from vad_turn_taking.backchannel import (
    backchannel_prediction_events,
    find_backchannel_ongoing,
    recover_bc_prediction_negatives,
)
from vad_turn_taking.vad import DialogEvents

from vad_turn_taking.backchannels import Backhannels
from vad_turn_taking.hold_shifts import HoldShift, get_dialog_states, get_last_speaker


class TurnTakingEvents:
    def __init__(
        self,
        bc_idx,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        pre_active=0.5,
        bc_pre_silence=1.5,
        bc_post_silence=3,
        bc_max_active=2,
        bc_prediction_window=1,
        bc_target_duration=0.3,
        bc_neg_active=1,
        bc_neg_prefix=1,
        frame_hz=100,
    ):
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
        self.bc_prediction_window = time_to_frames(bc_prediction_window, frame_hz)
        self.bc_target_duration = time_to_frames(bc_target_duration, frame_hz)
        self.bc_neg_active_frames = time_to_frames(bc_neg_active, frame_hz)
        self.bc_neg_prefix_frames = time_to_frames(bc_neg_prefix, frame_hz)

        self.bc_idx = bc_idx

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

    @torch.no_grad()
    def __call__(self, vad, projection_idx):
        n_frames = projection_idx.shape[1]

        ret = {}
        ret["hold"], ret["shift"] = DialogEvents.on_silence(
            vad,
            start_pad=self.frame_start_pad,
            target_frames=self.frame_target_duration,
            horizon=self.frame_horizon,
            min_context=self.frame_min_context,
            min_duration=self.frame_min_duration,
        )

        ret["pre_hold"], ret["pre_shift"] = DialogEvents.get_active_pre_events(
            vad,
            ret["hold"],
            ret["shift"],
            start_pad=self.frame_start_pad,
            active_frames=self.frame_pre_active,
            min_context=self.frame_min_context,
        )

        bc_ongoing_pos, bc_ongoing_neg = find_backchannel_ongoing(
            vad,
            n_test_frames=self.bc_target_duration,
            pre_silence_frames=self.bc_pre_silence_frames,
            post_silence_frames=self.bc_post_silence_frames,
            max_active_frames=self.bc_max_active_frames,
            neg_active_frames=self.bc_neg_active_frames,
            neg_prefix_frames=self.bc_neg_prefix_frames,
            min_context_frames=self.frame_min_context,
            n_frames=n_frames,
        )
        ret["backchannel"] = bc_ongoing_pos  # can be thought of as SHIFTS
        ret["backchannel_neg"] = bc_ongoing_neg  # Can be thought of as HOLDS

        # ret["backchannel"] = find_isolated_activity_on_other_active(
        #     vad,
        #     pre_silence_frames=self.bc_pre_silence_frames,
        #     post_silence_frames=self.bc_post_silence_frames,
        #     max_active_frames=self.bc_max_active_frames,
        # )

        if self.bc_idx.device != projection_idx.device:
            self.bc_idx = self.bc_idx.to(projection_idx.device)

        ret["backchannel_prediction"] = backchannel_prediction_events(
            projection_idx,
            vad,
            self.bc_idx,
            prediction_window=self.bc_prediction_window,
            isolated=ret["backchannel"],
        )
        ret["backchannel_prediction_neg"] = recover_bc_prediction_negatives(
            bc_ongoing_neg, neg_bc_prediction_window=self.bc_prediction_window
        )

        for return_name, vector in ret.items():
            ret[return_name] = vector[:, :n_frames]

        return ret


class TurnTakingEvents2:
    def __init__(
        self,
        shift_onset_cond=1,
        shift_offset_cond=1,
        hold_onset_cond=1,
        hold_offset_cond=1,
        min_silence=0.15,
        non_shift_horizon=2.0,
        non_shift_majority_ratio=0.95,
        metric_pad=0.05,
        metric_dur=0.1,
        metric_onset_dur=0.5,
        metric_pre_label_dur=0.5,
        bc_max_duration=0.5,
        bc_pre_silence=1.0,
        bc_post_silence=1.0,
        bc_metric_dur=0.5,
        frame_hz=100,
    ):

        assert (
            metric_onset_dur == bc_metric_dur
        ), "`metric_onset_dur` must be equal to `bc_metric_dur`"
        shift_onset_cond = time_to_frames(shift_onset_cond, frame_hz)
        shift_offset_cond = time_to_frames(shift_offset_cond, frame_hz)
        hold_onset_cond = time_to_frames(hold_onset_cond, frame_hz)
        hold_offset_cond = time_to_frames(hold_offset_cond, frame_hz)
        min_silence = time_to_frames(min_silence, frame_hz)
        metric_pad = time_to_frames(metric_pad, frame_hz)
        metric_dur = time_to_frames(metric_dur, frame_hz)
        metric_pre_label_dur = time_to_frames(metric_pre_label_dur, frame_hz)
        metric_onset_dur = time_to_frames(metric_onset_dur, frame_hz)
        non_shift_horizon = time_to_frames(non_shift_horizon, frame_hz)
        non_shift_majority_ratio = non_shift_majority_ratio

        # bc
        bc_max_duration_frames = time_to_frames(bc_max_duration, frame_hz)
        bc_pre_silence_frames = time_to_frames(bc_pre_silence, frame_hz)
        bc_post_silence_frames = time_to_frames(bc_post_silence, frame_hz)
        bc_metric_dur_frames = time_to_frames(bc_metric_dur, frame_hz)

        self.HS = HoldShift(
            shift_onset_cond=shift_onset_cond,
            shift_offset_cond=shift_offset_cond,
            hold_onset_cond=hold_onset_cond,
            hold_offset_cond=hold_offset_cond,
            min_silence=min_silence,
            metric_pad=metric_pad,
            metric_dur=metric_dur,
            metric_pre_label_dur=metric_pre_label_dur,
            metric_onset_dur=metric_onset_dur,
            non_shift_horizon=non_shift_horizon,
            non_shift_majority_ratio=non_shift_majority_ratio,
        )
        self.BS = Backhannels(
            max_duration_frames=bc_max_duration_frames,
            pre_silence_frames=bc_pre_silence_frames,
            post_silence_frames=bc_post_silence_frames,
            metric_dur_frames=bc_metric_dur_frames,
        )

    def __repr__(self):
        s = "TurnTakingEvents\n"
        s += str(self.HS) + "\n"
        s += str(self.BS)
        return s

    def __call__(self, vad):
        ds = get_dialog_states(vad)
        last_speaker = get_last_speaker(vad, ds)

        # Where a single speaker is active
        activity = ds == 0  # only A
        activity = torch.logical_or(activity, ds == 3)  # AND only B

        # HOLDS/SHIFTS
        # shift, pre_shift, long_shift_onset,
        # hold, pre_hold, long_hold_onset,
        # shift_overlap, pre_shift_overlap, non_shift

        tt = self.HS(vad, ds=ds)

        # Backchannels: backchannel
        bcs = self.BS(vad, last_speaker)

        tt.update(bcs)

        n = tt["non_shift"].shape[1]
        non_shift_on_activity = torch.logical_and(
            tt["non_shift"], activity[:, :n].unsqueeze(-1)
        )
        nons_act = torch.where(non_shift_on_activity)

        # LONG/SHORT
        # shorts: backchannels

        # Shift/Hold

        # Predict shift
        # Pos: window, on activity, prior to EOT before a SHIFT
        # Neg: Sampled from NON-SHIFT, on activity.

        # Predict backchannels
        # Pos: 1 second prior a backchannel
        # Neg: Sampled from NON-SHIFT, everywhere

        return tt


def debug_tt2():

    import matplotlib.pyplot as plt
    from conv_ssl.evaluation.utils import load_dm
    from vad_turn_taking.plot_utils import plot_vad_oh

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm(batch_size=16)
    diter = iter(dm.val_dataloader())
    batch = next(diter)

    eventer = TurnTakingEvents2(
        min_silence=0.2,
        metric_pad=0.1,
        metric_dur=0.1,
        metric_onset_dur=0.5,
        bc_metric_dur=0.5,
    )
    eventer

    batch = next(diter)

    vad = batch["vad"]
    tt = eventer(vad)

    _ = [print(k) for k in list(tt.keys())]

    n_rows = 4
    n_cols = 4
    fig, ax = plt.subplots(n_rows, n_cols, sharey=True, sharex=True, figsize=(16, 4))
    b = 0
    for row in range(n_rows):
        for col in range(n_cols):
            _ = plot_vad_oh(vad[b], ax=ax[row, col])
            _ = plot_vad_oh(
                tt["shift"][b], ax=ax[row, col], colors=["g", "g"], alpha=0.5
            )
            _ = plot_vad_oh(
                tt["shift_overlap"][b],
                ax=ax[row, col],
                colors=["darkgreen", "darkgreen"],
                alpha=0.8,
            )
            _ = plot_vad_oh(
                tt["hold"][b], ax=ax[row, col], colors=["r", "r"], alpha=0.5
            )
            _ = plot_vad_oh(
                tt["backchannel"][b],
                ax=ax[row, col],
                colors=["purple", "purple"],
                alpha=0.8,
            )
            _ = plot_vad_oh(
                tt["non_shift"][b].flip(-1),
                ax=ax[row, col],
                colors=["darkred", "darkred"],
                alpha=0.15,
            )
            b += 1
            if b == vad.shape[0]:
                break
        if b == vad.shape[0]:
            break
    plt.pause(0.1)


if __name__ == "__main__":

    from conv_ssl.evaluation.utils import load_dm
    from vad_turn_taking.plot_utils import plot_backchannel_prediction
    from vad_turn_taking.vad_projection import ProjectionCodebook, VadLabel
    from vad_turn_taking.backchannel import find_isolated_within
    import matplotlib.pyplot as plt

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm()
    # diter = iter(dm.val_dataloader())

    ####################################################################
    # Class to extract VAD-PROJECTION-WINDOWS from VAD
    bin_times = [0.2, 0.4, 0.6, 0.8]
    vad_hz = 100
    VL = VadLabel(bin_times=bin_times, vad_hz=vad_hz)
    # Codebook to extract specific class labels from onehot-representation
    codebook = ProjectionCodebook(bin_times=bin_times, frame_hz=vad_hz)

    ###################################################
    # Event extractor
    ###################################################
    metric_kwargs = {
        "event_pre": 0.5,
        "event_min_context": 1.0,
        "event_min_duration": 0.15,
        "event_horizon": 1.0,
        "event_start_pad": 0.05,
        "event_target_duration": 0.10,
        "event_bc_target_duration": 0.25,
        "event_bc_pre_silence": 1,
        "event_bc_post_silence": 1,
        "event_bc_max_active": 1.0,
        "event_bc_prediction_window": 0.4,
        "event_bc_neg_active": 1,
        "event_bc_neg_prefix": 1,
    }
    eventer = TurnTakingEvents(
        bc_idx=codebook.bc_prediction,
        horizon=metric_kwargs["event_horizon"],
        min_context=metric_kwargs["event_min_context"],
        start_pad=metric_kwargs["event_start_pad"],
        target_duration=metric_kwargs["event_target_duration"],
        pre_active=metric_kwargs["event_pre"],
        bc_target_duration=metric_kwargs["event_bc_target_duration"],
        bc_pre_silence=metric_kwargs["event_bc_pre_silence"],
        bc_post_silence=metric_kwargs["event_bc_post_silence"],
        bc_max_active=metric_kwargs["event_bc_max_active"],
        bc_prediction_window=metric_kwargs["event_bc_prediction_window"],
        bc_neg_active=metric_kwargs["event_bc_neg_active"],
        bc_neg_prefix=metric_kwargs["event_bc_neg_prefix"],
        frame_hz=vad_hz,
    )

    ###################################################
    # Batch
    ###################################################
    diter = iter(dm.val_dataloader())

    # for batch in diter:
    batch = next(diter)
    projection_idx = codebook(VL.vad_projection(batch["vad"]))
    vad = batch["vad"]
    isolated = find_isolated_within(
        vad,
        prefix_frames=eventer.bc_pre_silence_frames,
        max_duration_frames=eventer.bc_max_active_frames,
        suffix_frames=eventer.bc_post_silence_frames,
    )
    events = eventer(vad, projection_idx)
    # for k, v in events.items():
    #     print(f"{k}: {v.shape}")
    # Plot
    fig, ax = plot_backchannel_prediction(
        vad, events["backchannel"], bc_color="g", plot=False
    )
    for i, a in enumerate(ax):
        a.plot(events["backchannel_neg"][i, :, 0], color="r", linewidth=3)
        a.plot(-events["backchannel_neg"][i, :, 1], color="r", linewidth=3)
        a.plot(
            events["backchannel_prediction"][i, :, 0],
            color="g",
            linewidth=3,
            linestyle="dashed",
        )
        a.plot(
            -events["backchannel_prediction"][i, :, 1],
            color="g",
            linewidth=3,
            linestyle="dashed",
        )
        a.plot(
            events["backchannel_prediction_neg"][i, :, 0],
            color="r",
            linewidth=3,
            linestyle="dashed",
        )
        a.plot(
            -events["backchannel_prediction_neg"][i, :, 1],
            color="r",
            linewidth=3,
            linestyle="dashed",
        )
        a.plot(isolated[i, :, 0], color="k", linewidth=1)
        a.plot(-isolated[i, :, 1], color="k", linewidth=1)
    # plt.show()
    plt.pause(0.1)
