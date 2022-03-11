import torch

from vad_turn_taking.utils import find_island_idx_len, time_to_frames
from vad_turn_taking.backchannel import (
    backchannel_prediction_events,
    find_backchannel_ongoing,
    recover_bc_prediction_negatives,
)
from vad_turn_taking.vad import DialogEvents, VAD


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
