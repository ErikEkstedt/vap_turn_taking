import torch

from vad_turn_taking.utils import find_island_idx_len, time_to_frames
from vad_turn_taking.backchannel import (
    backchannel_prediction_events,
    find_isolated_activity_on_other_active,
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
        "event_bc_pre_silence": 1,
        "event_bc_post_silence": 2,
        "event_bc_max_active": 1,
        "event_bc_prediction_window": 0.5,
    }
    eventer = TurnTakingEvents(
        bc_idx=codebook.bc_prediction,
        horizon=metric_kwargs["event_horizon"],
        min_context=metric_kwargs["event_min_context"],
        start_pad=metric_kwargs["event_start_pad"],
        target_duration=metric_kwargs["event_target_duration"],
        pre_active=metric_kwargs["event_pre"],
        bc_pre_silence=metric_kwargs["event_bc_pre_silence"],
        bc_post_silence=metric_kwargs["event_bc_post_silence"],
        bc_max_active=metric_kwargs["event_bc_max_active"],
        bc_prediction_window=metric_kwargs["event_bc_prediction_window"],
        frame_hz=vad_hz,
    )

    # find bc-prediction-negatives
    def find_bc_prediction_negatives(vad, projection_window, ipu_lims):
        def get_cand_ipu(s, d):
            longer = d >= ipu_lims[0]
            if longer.sum() == 0:
                return None, None

            d = d[longer]
            s = s[longer]
            shorter = d <= ipu_lims[1]
            if shorter.sum() == 0:
                return None, None

            d = d[shorter]
            s = s[shorter]
            return s, d
            


        ds = VAD.vad_to_dialog_vad_states(vad)
        only_a = (ds == 0) * 1.
        only_b = (ds == 3) * 1.

        other_a = torch.logical_or(only_b, ds==2) * 1.
        other_b = torch.logical_or(only_a, ds==2) * 1.

        negs = torch.zeros_like(vad)

        for b in range(vad.shape[0]):
            break

            s1, d1, v1 = find_island_idx_len(only_a[b])
            s1 = s1[v1==1]
            d1 = d1[v1==1]
            e1 = s1 + d1
            s1_cand, d1_cand = get_cand_ipu(s1, d1)
            if s1_cand is not None:
                so, do, vo = find_island_idx_len(other_a[b])
                so = so[vo==1]
                do = do[vo==1]
                eo = so + do
            


            s2, d2, v2 = find_island_idx_len(only_b[b])
            s2 = s2[v2==1]
            d2 = d2[v2==1]
            s2_cand, d2_cand = get_cand_ipu(s2, d2)
            e2_cand = s2_cand + d2_cand

            if s2_cand is not None:
                so, do, vo = find_island_idx_len(other_b[b])
                so = so[vo==1]
                do = do[vo==1]
                eo = so + do

                cands2 = []
                for s_cand, d_cand in zip(s2_cand, d2_cand):

                    for sother, eother in zip(so, eo):

                        if s_cand < sother and e_cand < sother:
                            e_cand = s_cand + d_cand
                            if e_cand-projection_window > 0:
                                negs[b, e_cand-projection_window:e_cand, 1] = 1.
                        elif sother < s_cand and seother<e_cand < sother: # candidate before other
                            if e-projection_window > 0:
                                negs[b, e-projection_window:e, 1] = 1.

                        # sdiff = sother - s
                        # if sdiff < 0:  # other before cand
                        #     ediff = eother - s
                        #     if ediff < 0: # other end before cand
                        #         pass
                        # else: # cand before other
                        #     pass

                            # if ediff
            # for start, dur in zip(s, d):
            #     end = start+dur
            #     start = end - projection_window
            #     if start > 0:
            #         negs[b, start:end] = 1.
        return negs


    projection_window=50
    ipu_lims = [200, 400]

    ###################################################
    # Batch
    ###################################################
    diter = iter(dm.val_dataloader())
    batch = next(diter)

    # batch = next(iter(dm.val_dataloader()))
    projection_idx = codebook(VL.vad_projection(batch["vad"]))
    vad = batch["vad"]
    print("projection_idx: ", tuple(projection_idx.shape))
    print("vad: ", tuple(vad.shape))
    events = eventer(vad, projection_idx)
    for k, v in events.items():
        print(f"{k}: {v.shape}")

    # Plot
    # Find single speaker
    # negs = find_bc_prediction_negatives(vad, projection_window, ipu_lims)
    negs = events['backchannel_prediction'][:, 100:]
    negs = torch.cat((negs, torch.zeros((4, 100, 2))), dim=1)
    negs[:, :100] = 0
    fig, ax = plot_backchannel_prediction(
        vad, events["backchannel_prediction"], bc_color='g', plot=True
    )
    for i, a in enumerate(ax):
        # a.plot(only_a[i]*.5, color='orange', linewidth=2)
        # a.plot(-only_b[i]*.5, color='blue', linewidth=2)
        a.plot(negs[i, :, 0], color='r', linewidth=3)
        a.plot(-negs[i, :, 1], color='r', linewidth=3)
    plt.pause(0.1)


