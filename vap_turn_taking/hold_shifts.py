import torch
from typing import Dict, List, Tuple

from vap_turn_taking.utils import time_to_frames
import vap_turn_taking.functional as VF


class HoldShift:
    """
    Hold/Shift extraction from VAD. Operates of Frames.

        Arguments:
            post_onset_shift:           int, frames for shift onset cond
            pre_offset_shift:           int, frames for shift offset cond
            post_onset_hold:            int, frames for hold onset cond
            pre_offset_hold:            int, frames for hold offset cond
            metric_pad:                 int, pad on silence (shift/hold) onset used for evaluating
            metric_dur:                 int, duration off silence (shift/hold) used for evaluating
            metric_pre_label_dur:       int, frames prior to Shift-silence for prediction on-active shift
            non_shift_horizon:          int, frames to define majority speaker window for Non-shift
            non_shift_majority_ratio:   float, ratio of majority speaker

        Return:
            dict:       {'shift', 'pre_shift', 'hold', 'pre_hold', 'non_shift'}

        Active: "---"
        Silent: "..."

        # SHIFTS

        onset:                                 |<-- only A -->|
        A:          ...........................|-------------------
        B:          ----------------|..............................
        offset:     |<--  only B -->|
        SHIFT:                      |XXXXXXXXXX|

        -----------------------------------------------------------
        # HOLDS

        onset:                                 |<-- only B -->|
        A:          ...............................................
        B:          ----------------|..........|-------------------
        offset:     |<--  only B -->|
        HOLD:                       |XXXXXXXXXX|

        -----------------------------------------------------------
        # NON-SHIFT

        Horizon:                        |<-- B majority -->|
        A:          .....................................|---------
        B:          ----------------|......|------|................
        non_shift:  |XXXXXXXXXXXXXXXXXXX|

        A future horizon window must contain 'majority' activity from
        from the last speaker. In these moments we "know" a shift
        is a WRONG prediction. But closer to activity from the 'other'
        speaker, a turn-shift is appropriate.

        -----------------------------------------------------------
        # metrics
        e.g. shift

        onset:                                     |<-- only A -->|
        A:          ...............................|---------------
        B:          ----------------|..............................
        offset:     |<--  only B -->|
        SHIFT:                      |XXXXXXXXXXXXXX|
        metric:                     |...|XXXXXX|
        metric:                     |pad|  dur |

        -----------------------------------------------------------


        Using 'dialog states' consisting of 4 different states
        0. Only A is speaking
        1. Silence
        2. Overlap
        3. Only B is speaking

        Shift GAP:       0 -> 1 -> 3          3 -> 1 -> 0
        Shift Overlap:   0 -> 2 -> 3          3 -> 2 -> 0
        HOLD:            0 -> 1 -> 0          3 -> 1 -> 3
    """

    def __init__(
        self,
        post_onset_shift,
        pre_offset_shift,
        post_onset_hold,
        pre_offset_hold,
        non_shift_horizon,
        metric_pad,
        metric_dur,
        metric_pre_label_dur,
        metric_onset_dur,
        non_shift_majority_ratio=1,
    ):
        assert (
            metric_onset_dur <= post_onset_shift
        ), "`metric_onset_dur` must be less or equal to `post_onset_shift`"

        self.post_onset_shift = post_onset_shift
        self.pre_offset_shift = pre_offset_shift
        self.post_onset_hold = post_onset_hold
        self.pre_offset_hold = pre_offset_hold

        self.metric_pad = metric_pad
        self.metric_dur = metric_dur
        self.min_silence = metric_pad + metric_dur
        self.metric_pre_label_dur = metric_pre_label_dur
        self.metric_onset_dur = metric_onset_dur

        self.non_shift_horizon = non_shift_horizon
        self.non_shift_majority_ratio = non_shift_majority_ratio

        # Templates
        self.shift_template = torch.tensor([[3, 1, 0], [0, 1, 3]])  # on Silence
        self.shift_overlap_template = torch.tensor([[3, 2, 0], [0, 2, 3]])
        self.hold_template = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence

    def __repr__(self):
        s = "Holds & Shifts"
        s += f"\n  post_onset_shift: {self.post_onset_shift}"
        s += f"\n  pre_offset_shift: {self.pre_offset_shift}"
        s += f"\n  post_onset_hold: {self.post_onset_hold}"
        s += f"\n  pre_offset_hold: {self.pre_offset_hold}"
        s += f"\n  min_silence: {self.min_silence}"
        s += f"\n  metric_pad: {self.metric_pad}"
        s += f"\n  metric_dur: {self.metric_dur}"
        s += f"\n  metric_pre_label_dur: {self.metric_pre_label_dur}"
        s += f"\n  non_shift_horizon: {self.non_shift_horizon}"
        s += f"\n  non_shift_majority_ratio: {self.non_shift_majority_ratio}"
        return s

    def fill_template(self, vad, ds, template):
        """
        Used in practice to create VAD -> FILLED_VAD, where filled vad combines
        consecutive segments of activity from the same speaker as a single
        chunk.
        """

        filled_vad = vad.clone()
        for b in range(ds.shape[0]):
            s, d, v = VF.find_island_idx_len(ds[b])
            if len(v) < 3:
                continue
            triads = v.unfold(0, size=3, step=1)
            next_speaker, steps = torch.where(
                (triads == template.unsqueeze(1)).sum(-1) == 3
            )
            for ns, pre in zip(next_speaker, steps):
                cur = pre + 1
                # Fill the matching template
                filled_vad[b, s[cur] : s[cur] + d[cur], ns] = 1.0
        return filled_vad

    def match_template(
        self,
        vad,
        ds,
        template,
        pre_cond_frames,
        post_cond_frames,
        pre_match=False,
        onset_match=False,
        max_frame=None,
        min_context=0,
    ):
        """
        Creates a onehot vector where the steps matching the template.
        Return:
            match_oh:       torch.Tensor (B, N, 2), where the last bin corresponds to the next speaker
        """

        hold_cond = template[0, 0] == template[0, -1]

        match_oh = torch.zeros((*ds.shape, 2), device=ds.device, dtype=torch.float)

        pre_match_oh = None
        if pre_match:
            pre_match_oh = torch.zeros(
                (*ds.shape, 2), device=ds.device, dtype=torch.float
            )

        onset_match_oh = None
        if onset_match:
            onset_match_oh = torch.zeros(
                (*ds.shape, 2), device=ds.device, dtype=torch.float
            )
        for b in range(ds.shape[0]):
            s, d, v = VF.find_island_idx_len(ds[b])

            if len(v) < 3:
                continue

            triads = v.unfold(0, size=3, step=1)
            next_speaker, steps = torch.where(
                (triads == template.unsqueeze(1)).sum(-1) == 3
            )

            # ns: next_speaker, pre_step
            for ns, pre_step in zip(next_speaker, steps):
                # If template is of 'HOLD-type' then previous speaker is the
                # same as next speaker. Otherwise they are different.
                nos = 0 if ns == 1 else 1  # strictly the OTHER 'next speaker'
                ps = ns if hold_cond else nos  # previous speaker

                cur = pre_step + 1
                post = pre_step + 2

                # Silence Condition: if the current step is silent (shift with gap and holds)
                # then we only care about silences over a certain duration.
                if v[cur] == 1 and d[cur] < self.min_silence:
                    continue

                # Can this be useful? older way of only considering active segments where
                # pauses have not been filled...
                # Shifts are more sensible to overall activity around silence/overlap
                # and uses `filled_vad` as vad where consecutive
                # if vad is None:
                #     if d[pre_step] >= pre_cond_frames and d[post] >= post_cond_frames:
                #         match_oh[b, s[cur] : s[cur] + d[cur], ns] = 1.0
                #     continue

                # pre_condition
                # using a filled version of the VAD signal we check wheather
                # only the 'previous speaker, ps' was active. This will then include
                # activity from that speaker deliminated by silence/pauses/holds
                pre_start = s[cur] - pre_cond_frames

                # print('pre_start: ', pre_start, s[cur])
                pre_cond1 = vad[b, pre_start : s[cur], ps].sum() == pre_cond_frames
                not_ps = 0 if ps == 1 else 1
                pre_cond2 = vad[b, pre_start : s[cur], not_ps].sum() == 0
                pre_cond = torch.logical_and(pre_cond1, pre_cond2)

                if not pre_cond:
                    # pre_cond = vad[b, pre_start : s[cur], ps].sum()
                    # print("pre cond Failed: ", pre_cond, pre_cond_frames)
                    # # print(vad[b, pre_start:s[cur]+d[cur]+10])
                    # input()
                    continue

                # single speaker post
                post_start = s[post]
                post_end = post_start + post_cond_frames
                post_cond1 = vad[b, post_start:post_end, ns].sum() == post_cond_frames
                post_cond2 = vad[b, post_start:post_end, nos].sum() == 0
                post_cond = torch.logical_and(post_cond1, post_cond2)
                if not post_cond:
                    # post_cond = vad[b, post_start:post_end, ns].sum()
                    # print("post cond Failed: ", post_cond, post_cond_frames)
                    # print(vad[b, pre_start:s[cur]+d[cur]+10])
                    # input()
                    continue

                # start = s[cur]
                # end = s[cur] + d[cur]
                # if self.metric_pad > 0:
                #     start += self.metric_pad
                #
                # if self.metric_dur > 0:
                #     end = start + self.metric_dur

                # Max frame condition:
                # Can't have event outside of predictable window
                if max_frame is not None:
                    if s[cur] >= max_frame:
                        continue

                # Min context condition:
                if (s[cur] + self.metric_pad) < min_context:
                    continue

                if pre_match:
                    pre_match_oh[
                        b, s[cur] - self.metric_pre_label_dur : s[cur], ns
                    ] = 1.0

                # end = s[cur] + self.metric_pad + d[cur]
                end = s[cur] + self.metric_pad + self.metric_dur
                # Max frame condition:
                # Can't have event outside of predictable window
                if max_frame is not None:
                    if end >= max_frame:
                        continue

                match_oh[b, s[cur] + self.metric_pad : end, ns] = 1.0

                if onset_match:
                    end = s[post] + self.metric_onset_dur
                    if max_frame is not None:
                        if end >= max_frame:
                            continue
                    onset_match_oh[b, s[post] : end, ns] = 1.0

        return match_oh, pre_match_oh, onset_match_oh

    def non_shifts(
        self,
        vad,
        last_speaker,
        horizon,
        majority_ratio=1,
        max_frame=None,
        min_context=0,
    ):
        """

        Non-shifts are all parts of the VAD signal where a future of `horizon`
        frames "overwhelmingly" belongs to a single speaker. The
        `majority_ratio` is a threshold over which the ratio of activity must belong to the last/current-speaker.

        Arguments:
            vad:                torch.Tensor, (B, N, 2)
            horizon:            int, length in frames of the horizon
            majority_ratio:     float, ratio of which the majority speaker must occupy
        """

        EPS = 1e-5  # used to avoid nans

        nb = vad.size(0)

        # future windows
        vv = vad[:, 1:].unfold(1, size=horizon, step=1).sum(dim=-1)
        vv = vv / (vv.sum(-1, keepdim=True) + EPS)

        diff = vad.shape[1] - vv.shape[1]

        if max_frame is not None:
            vv = vv[:, :max_frame]

        # Majority_ratio. Add eps to value to not miss majority_ratio==1.
        # because we divided 1
        maj_speaker_cond = majority_ratio <= (vv + EPS)

        # Last speaker
        a_last = last_speaker[:, : maj_speaker_cond.shape[1]] == 0
        b_last = last_speaker[:, : maj_speaker_cond.shape[1]] == 1
        a_non_shift = torch.logical_and(a_last, maj_speaker_cond[..., 0])
        b_non_shift = torch.logical_and(b_last, maj_speaker_cond[..., 1])
        ns = torch.stack((a_non_shift, b_non_shift), dim=-1).float()
        # fill to correct size (same as vad and all other events)
        z = torch.zeros((nb, diff, 2), device=ns.device)
        non_shift = torch.cat((ns, z), dim=1)

        # Min Context Condition
        # i.e. don't use negatives from before `min_context`
        if min_context > 0:
            non_shift[:, :min_context] = 0.0
        return non_shift

    def __call__(
        self,
        vad,
        ds=None,
        filled_vad=None,
        max_frame=None,
        min_context=0,
        return_list=False,
    ):

        if ds is None:
            ds = VF.get_dialog_states(vad)

        if vad.device != self.hold_template.device:
            self.shift_template = self.shift_template.to(vad.device)
            self.shift_overlap_template = self.shift_overlap_template.to(vad.device)
            self.hold_template = self.hold_template.to(vad.device)

        if filled_vad is None:
            filled_vad = self.fill_template(vad, ds, self.hold_template)

        shift_oh, pre_shift_oh, long_shift_onset = self.match_template(
            filled_vad,
            ds,
            self.shift_template,
            pre_cond_frames=self.pre_offset_shift,
            post_cond_frames=self.post_onset_shift,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
            min_context=min_context,
        )
        shift_ov_oh, _, _ = self.match_template(
            filled_vad,
            ds,
            self.shift_overlap_template,
            pre_cond_frames=self.pre_offset_shift,
            post_cond_frames=self.post_onset_shift,
            pre_match=False,
            onset_match=False,
            max_frame=max_frame,
            min_context=min_context,
        )
        hold_oh, pre_hold_oh, long_hold_onset = self.match_template(
            filled_vad,
            ds,
            self.hold_template,
            pre_cond_frames=self.pre_offset_hold,
            post_cond_frames=self.post_onset_hold,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
            min_context=min_context,
        )

        from vap_turn_taking.utils import get_last_speaker

        last_speaker = get_last_speaker(vad, ds)
        non_shift_oh = self.non_shifts(
            vad,
            last_speaker,
            horizon=self.non_shift_horizon,
            majority_ratio=self.non_shift_majority_ratio,
            max_frame=max_frame,
            min_context=min_context,
        )

        return {
            "shift": shift_oh,
            "pre_shift": pre_shift_oh,
            "long_shift_onset": long_shift_onset,
            "hold": hold_oh,
            "pre_hold": pre_hold_oh,
            "long_hold_onset": long_hold_onset,
            "shift_overlap": shift_ov_oh,
            "non_shift": non_shift_oh,
        }


class HoldShiftNew:
    def __init__(
        self,
        pre_cond_time: float = 1,
        post_cond_time: float = 1,
        min_silence_time: float = 0.2,
        min_context_time: float = 3,
        max_time: int = 10,
        frame_hz: int = 50,
    ):

        # Time
        self.pre_cond_time = pre_cond_time
        self.post_cond_time = post_cond_time
        self.min_silence_time = min_silence_time
        self.min_context_time = min_context_time
        self.max_time = max_time

        # Frames
        self.pre_cond_frame = time_to_frames(pre_cond_time, frame_hz)
        self.post_cond_frame = time_to_frames(post_cond_time, frame_hz)
        self.min_silence_frame = time_to_frames(min_silence_time, frame_hz)
        self.min_context_frame = time_to_frames(min_context_time, frame_hz)
        self.max_frame = time_to_frames(max_time, frame_hz)

    def __repr__(self) -> str:
        s = "HoldShift"
        s += "\n---------"
        s += f"\n  Time:"
        s += f"\n\tpre_cond_time     = {self.pre_cond_time}s"
        s += f"\n\tpost_cond_time    = {self.post_cond_time}s"
        s += f"\n\tmin_silence_time  = {self.min_silence_time}s"
        s += f"\n\tmin_context_time  = {self.min_context_time}s"
        s += f"\n\tmax_time          = {self.max_time}s"
        s += f"\n  Frame:"
        s += f"\n\tpre_cond_frame    = {self.pre_cond_frame}"
        s += f"\n\tpost_cond_frame   = {self.post_cond_frame}"
        s += f"\n\tmin_silence_frame = {self.min_silence_frame}"
        s += f"\n\tmin_context_frame = {self.min_context_frame}"
        s += f"\n\tmax_frame         = {self.max_frame}"
        return s

    def __call__(self, vad: torch.Tensor) -> Dict[str, List[List[Tuple[int, int]]]]:
        assert (
            vad.ndim == 3
        ), f"Expected vad.ndim=3 (B, N_FRAMES, 2) but got {vad.shape}"

        batch_size = vad.shape[0]

        shifts, holds = [], []
        for b in range(batch_size):
            ds = VF.get_dialog_states(vad[b])
            tmp_shifts, tmp_holds = VF.hold_shift_regions(
                vad=vad[b],
                ds=ds,
                pre_cond_frames=self.pre_cond_frame,
                post_cond_frames=self.post_cond_frame,
                min_silence_frames=self.min_silence_frame,
                min_context_frames=self.min_context_frame,
                max_frame=self.max_frame,
            )
            shifts.append(tmp_shifts)
            holds.append(tmp_holds)
        return {"shift": shifts, "hold": holds}


def _old_main():
    import matplotlib.pyplot as plt
    from vap_turn_taking.plot_utils import plot_vad_oh, plot_event
    from vap_turn_taking.config.example_data import event_conf_frames, example

    plt.close("all")

    hs_kwargs = event_conf_frames["hs"]
    HS = HoldShift(**hs_kwargs)
    tt = HS(example["va"], max_frame=None)
    for k, v in tt.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {tuple(v.shape)}")
        else:
            print(f"{k}: {v}")
    print("shift: ", (example["shift"] != tt["shift"]).sum())
    print("hold: ", (example["hold"] != tt["hold"]).sum())

    fig, ax = plot_vad_oh(va[0])
    # # _, ax = plot_event(tt["shift"][0], ax=ax)
    # _, ax = plot_event(s[0], color=["g", "g"], ax=ax)
    # _, ax = plot_event(h[0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(bc[0], color=["b", "b"], ax=ax)
    # _, ax = plot_event(tt["shift_overlap"][0], ax=ax)
    # _, ax = plot_event(tt_bc["backchannel"][0], color=["b", "b"], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt_bc["pre_backchannel"][0], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt["hold"][0], color=["r", "r"], ax=ax)
    # _, ax = plot_event(tt['pre_shift'][0], color=['g', 'g'], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt['pre_hold'][0], color=['r', 'r'], alpha=0.2, ax=ax)
    # _, ax = plot_event(tt['long_shift_onset'][0], color=['r', 'r'], alpha=0.2, ax=ax)
    _, ax = plot_event(tt["non_shift"][0], color=["r", "r"], alpha=0.2, ax=ax)
    plt.pause(0.1)


def _time_comparison():
    import timeit
    from vap_turn_taking.config.example_data import event_conf_frames

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    vad = torch.cat([vad] * 10)

    hs_kwargs = event_conf_frames["hs"]
    HS_OLD = HoldShift(**hs_kwargs)
    HS = HoldShiftNew()
    old = timeit.timeit("HS_OLD(vad)", globals=globals(), number=200)
    new = timeit.timeit("HS(vad)", globals=globals(), number=200)
    print("Old: ", old)
    print("New: ", new)
    if old > new:
        print(f"NEW approach is {old/new} times faster!")
    else:
        print(f"OLD approach is {new/old} times faster!")


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from vap_turn_taking.plot_utils import plot_vad_oh

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    batch_size = vad.shape[0]

    HS = HoldShiftNew()
    sh_events = HS(vad)

    for b in range(batch_size):
        shifts = sh_events["shift"][b]
        holds = sh_events["hold"][b]
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
        _ = plot_vad_oh(vad[b], ax=ax)
        ax.axvline(HS.min_context_frame, linewidth=4, color="k")
        ax.axvline(HS.max_frame, linewidth=4, color="k")
        for start, end in shifts:
            ax.axvline(start, linewidth=2, color="g")
            ax.axvline(end, linewidth=2, color="r")
        for start, end in holds:
            ax.axvline(start, linewidth=2, linestyle="dashed", color="g")
            ax.axvline(end, linewidth=2, linestyle="dashed", color="r")
        plt.show()
