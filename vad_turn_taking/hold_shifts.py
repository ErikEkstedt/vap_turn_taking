import torch

from vad_turn_taking.utils import find_island_idx_len


def get_dialog_states(vad) -> torch.Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def last_speaker_single(s):
    start, _, val = find_island_idx_len(s)

    # exlude silences (does not effect last_speaker)
    # silences should be the value of the previous speaker
    sil_idx = torch.where(val == 1)[0]
    if len(sil_idx) > 0:
        if sil_idx[0] == 0:
            val[0] = 2  # 2 is both we don't know if its a shift or hold
            sil_idx = sil_idx[1:]
        val[sil_idx] = val[sil_idx - 1]
    # map speaker B state (=3) to 1
    val[val == 3] = 1
    # get repetition lengths
    repeat = start[1:] - start[:-1]
    # Find difference between original and repeated
    # and use diff to repeat the last speaker until the end of segment
    diff = len(s) - repeat.sum(0)
    repeat = torch.cat((repeat, diff.unsqueeze(0)))
    # repeat values to create last speaker over entire segment
    last_speaker = torch.repeat_interleave(val, repeat)
    return last_speaker


def get_last_speaker(vad, ds):
    assert vad.ndim > 1, "must provide vad of size: (N, channels) or (B, N, channels)"

    # get last active speaker (for turn shift/hold)
    if vad.ndim < 3:
        last_speaker = last_speaker_single(ds)
    else:  # (B, N, Channels) = (B, N, n_speakers)
        last_speaker = []
        for b in range(vad.shape[0]):
            s = ds[b]
            last_speaker.append(last_speaker_single(s))
        last_speaker = torch.stack(last_speaker)
    return last_speaker


class HoldShift:
    """
    Hold/Shift extraction from VAD. Operates of Frames.

    Arguments:
        shift_onset_cond:           int, frames for shift onset cond
        shift_offset_cond:          int, frames for shift offset cond
        hold_onset_cond:            int, frames for hold onset cond
        hold_offset_cond:           int, frames for hold offset cond
        min_silence:                int, frames defining the minimimum amount of silence for Hold/Shift
        metric_pad:                 int, pad on silence (shift/hold) onset used for evaluating
        metric_dur:                 int, duration off silence (shift/hold) used for evaluating
        pre_label_frames:           int, frames prior to Shift-silence for prediction on-active shift
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
        shift_onset_cond,
        shift_offset_cond,
        hold_onset_cond,
        hold_offset_cond,
        min_silence,
        metric_pad,
        metric_dur,
        metric_pre_label_dur,
        metric_onset_dur,
        non_shift_horizon,
        non_shift_majority_ratio,
    ):

        assert (
            metric_pad + metric_dur
        ) <= min_silence, (
            "the sum of `metric_pad` and `metric_dur` must be less than `min_silence`"
        )

        assert (
            metric_onset_dur <= shift_onset_cond
        ), "`metric_onset_dur` must be less or equal to `shift_onset_cond`"
        self.shift_onset_cond = shift_onset_cond
        self.shift_offset_cond = shift_offset_cond
        self.hold_onset_cond = hold_onset_cond
        self.hold_offset_cond = hold_offset_cond

        self.min_silence = min_silence
        self.metric_pad = metric_pad
        self.metric_dur = metric_dur
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
        s += f"\n  shift_onset_cond: {self.shift_onset_cond}"
        s += f"\n  shift_offset_cond: {self.shift_offset_cond}"
        s += f"\n  hold_onset_cond: {self.hold_onset_cond}"
        s += f"\n  hold_offset_cond: {self.hold_offset_cond}"
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
            s, d, v = find_island_idx_len(ds[b])
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
            s, d, v = find_island_idx_len(ds[b])

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

                if pre_match:
                    pre_match_oh[
                        b, s[cur] - self.metric_pre_label_dur : s[cur], ns
                    ] = 1.0

                end = s[cur] + self.metric_pad + d[cur]
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

    def non_shifts(self, vad, last_speaker, horizon, majority_ratio=1, max_frame=None):
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

        # future windows
        vv = vad[:, 1:].unfold(1, size=horizon, step=1).sum(dim=-1)
        vv = vv / (vv.sum(-1, keepdim=True) + EPS)

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
        non_shift = torch.stack((a_non_shift, b_non_shift), dim=-1).float()
        return non_shift

    def __call__(self, vad, ds=None, filled_vad=None, max_frame=1000):
        if ds is None:
            ds = get_dialog_states(vad)

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
            pre_cond_frames=self.shift_offset_cond,
            post_cond_frames=self.shift_onset_cond,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
        )
        shift_ov_oh, _, _ = self.match_template(
            filled_vad,
            ds,
            self.shift_overlap_template,
            pre_cond_frames=self.shift_offset_cond,
            post_cond_frames=self.shift_onset_cond,
            pre_match=False,
            onset_match=False,
            max_frame=max_frame,
        )
        hold_oh, pre_hold_oh, long_hold_onset = self.match_template(
            filled_vad,
            ds,
            self.hold_template,
            pre_cond_frames=self.hold_offset_cond,
            post_cond_frames=self.hold_onset_cond,
            pre_match=True,
            onset_match=True,
            max_frame=max_frame,
        )

        last_speaker = get_last_speaker(vad, ds)
        non_shift_oh = self.non_shifts(
            vad,
            last_speaker,
            horizon=self.non_shift_horizon,
            majority_ratio=self.non_shift_majority_ratio,
            max_frame=max_frame,
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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from vad_turn_taking.plot_utils import plot_vad_oh

    vad = torch.zeros((3, 150, 2), dtype=torch.float)
    # Gap shifts
    vad[0, :30, 1] = 1.0
    vad[0, 50:80, 0] = 1.0
    vad[0, 100:140, 1] = 1.0
    # Hold
    vad[1, :30, 1] = 1.0
    vad[1, 40:80, 1] = 1.0
    vad[1, 100:120, 1] = 1.0
    # Overlap shifts
    vad[2, :50, 1] = 1.0
    vad[2, 40:80, 0] = 1.0
    vad[2, 100:140, 1] = 1.0

    ds = get_dialog_states(vad)

    # HS = HoldShift(
    #     shift_onset_cond=30,
    #     shift_offset_cond=30,
    #     hold_onset_cond=20,
    #     hold_offset_cond=30,
    #     non_shift_horizon=20,
    #     non_shift_majority_ratio=0.9,
    # )
    # tt = HS(vad)
    # b = 0
    # fig, ax = plt.subplots(3, 1)
    # _ = plot_vad_oh(vad[b], ax=ax[0])
    # _ = plot_vad_oh(tt["shift"][b], ax=ax[1], colors=["g", "darkgreen"])
    # _ = plot_vad_oh(tt["non_shift"][b], ax=ax[2])
    # plt.pause(0.1)

    from conv_ssl.evaluation.utils import load_dm

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm(batch_size=16)
    diter = iter(dm.val_dataloader())

    batch = next(diter)

    vad = batch["vad"]
    ds = get_dialog_states(vad)
    last_speaker = get_last_speaker(vad, ds)

    HS = HoldShift(
        shift_onset_cond=100,
        shift_offset_cond=100,
        hold_onset_cond=40,
        hold_offset_cond=20,
        non_shift_horizon=200,
        non_shift_majority_ratio=0.95,
    )
    # ds = get_dialog_states(vad)
    # filled_vad = HS.fill_template(vad, ds, HS.hold_template)
    # b = 2
    # fig, ax = plt.subplots(2, 1, sharey=True, sharex=True, figsize=(16, 4))
    # _ = plot_vad_oh(vad[b], ax=ax[0])
    # _ = plot_vad_oh(filled_vad[b], ax=ax[1])
    # plt.show()
    tt = HS(vad)

    start = 0

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
