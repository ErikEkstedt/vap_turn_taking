import torch

from vad_turn_taking.utils import time_to_frames
from vad_turn_taking.backchannel import (
    backchannel_prediction_events,
    find_isolated_activity_on_other_active,
)
from vad_turn_taking.vad import DialogEvents


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

        ret["backchannel"] = find_isolated_activity_on_other_active(
            vad,
            pre_silence_frames=self.bc_pre_silence_frames,
            post_silence_frames=self.bc_post_silence_frames,
            max_active_frames=self.bc_max_active_frames,
        )

        if self.bc_idx.device != projection_idx.device:
            self.bc_idx = self.bc_idx.to(projection_idx.device)

        # print("proj_idx: ", projection_idx.device)
        # print("vad: ", vad.device)
        # print("bc_idx: ", self.bc_idx.device)
        ret["backchannel_prediction"] = backchannel_prediction_events(
            projection_idx,
            vad,
            self.bc_idx,
            prediction_window=self.bc_prediction_window,
            isolated=ret["backchannel"],
        )

        n_frames = projection_idx.shape[1]
        for return_name, vector in ret.items():
            ret[return_name] = vector[:, :n_frames]

        return ret


if __name__ == "__main__":

    from conv_ssl.evaluation.utils import load_dm
    from vad_turn_taking.plot_utils import plot_backchannel_prediction
    from vad_turn_taking.vad_projection import ProjectionCodebook, VadLabel

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm()
    diter = iter(dm.val_dataloader())

    ####################################################################
    # Class to extract VAD-PROJECTION-WINDOWS from VAD
    VL = VadLabel(bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=100, threshold_ratio=0.5)
    # Codebook to extract specific class labels from onehot-representation
    codebook = ProjectionCodebook()

    ###################################################
    # Event extractor
    ###################################################
    eventer = TurnTakingEvents(bc_idx=codebook.bc_active)

    ###################################################
    # Batch
    ###################################################
    batch = next(diter)
    vad = batch["vad"]

    batch = next(diter)
    projection_idx = codebook(VL.vad_projection(batch["vad"]))
    vad = batch["vad"]
    print("projection_idx: ", tuple(projection_idx.shape))
    print("vad: ", tuple(vad.shape))

    events = eventer(vad, projection_idx)
    for k, v in events.items():
        print(f"{k}: {v.shape}")

    # Predict upcomming Backchannel
    fig, ax = plot_backchannel_prediction(
        vad, events["backchannel_prediction"], plot=True
    )
