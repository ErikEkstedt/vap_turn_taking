import torch
import torch.nn.functional as F
from einops import rearrange


def time_to_frames(time, frame_hz):
    if isinstance(time, list):
        time = torch.tensor(time)

    frame = time * frame_hz

    if isinstance(frame, torch.Tensor):
        frame = frame.long().tolist()
    else:
        frame = int(frame)

    return frame


def frame2time(f, frame_time):
    return f * frame_time


def time2frames(t, hop_time):
    return int(t / hop_time)


def find_island_idx_len(x):
    """
    Finds patches of the same value.

    starts_idx, duration, values = find_island_idx_len(x)

    e.g:
        ends = starts_idx + duration

        s_n = starts_idx[values==n]
        ends_n = s_n + duration[values==n]  # find all patches with N value

    """
    assert x.ndim == 1
    n = len(x)
    y = x[1:] != x[:-1]  # pairwise unequal (string safe)
    i = torch.cat(
        (torch.where(y)[0], torch.tensor(n - 1, device=x.device).unsqueeze(0))
    ).long()
    it = torch.cat((torch.tensor(-1, device=x.device).unsqueeze(0), i))
    dur = it[1:] - it[:-1]
    idx = torch.cumsum(
        torch.cat((torch.tensor([0], device=x.device, dtype=torch.long), dur)), dim=0
    )[
        :-1
    ]  # positions
    return idx, dur, x[i]


def find_label_match(source_idx, target_idx):
    match = torch.where(source_idx.unsqueeze(-1) == target_idx)
    midx = target_idx[match[-1]]  # back to original idx
    frames = torch.zeros_like(source_idx)
    # Does not work on gpu: frames[match[:-1]] = 1.0
    frames[match[:-1]] = torch.ones_like(match[0])
    return frames, midx


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


def vad_list_to_onehot(vad_list, hop_time, duration, channel_last=False):
    n_frames = time2frames(duration, hop_time) + 1

    if isinstance(vad_list[0][0], list):
        vad_tensor = torch.zeros((len(vad_list), n_frames))
        for ch, ch_vad in enumerate(vad_list):
            for v in ch_vad:
                s = time2frames(v[0], hop_time)
                e = time2frames(v[1], hop_time)
                vad_tensor[ch, s:e] = 1.0
    else:
        vad_tensor = torch.zeros((1, n_frames))
        for v in vad_list:
            s = time2frames(v[0], hop_time)
            e = time2frames(v[1], hop_time)
            vad_tensor[:, s:e] = 1.0

    if channel_last:
        vad_tensor = vad_tensor.permute(1, 0)

    return vad_tensor


def vad_to_dialog_vad_states(vad) -> torch.Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def mutual_silences(vad, ds=None):
    if ds is None:
        ds = vad_to_dialog_vad_states(vad)
    return ds == 1


def get_current_vad_onehot(vad, end, duration, speaker, frame_size):
    """frame_size in seconds"""
    start = end - duration
    n_frames = int(duration / frame_size)
    vad_oh = torch.zeros((2, n_frames))

    for ch, ch_vad in enumerate(vad):
        for s, e in ch_vad:
            if start <= s <= end:
                rel_start = s - start
                v_start_frame = round(rel_start / frame_size)
                if start <= e <= end:  # vad segment completely in chunk
                    rel_end = e - start
                    v_end_frame = round(rel_end / frame_size)
                    vad_oh[ch, v_start_frame : v_end_frame + 1] = 1.0
                else:  # only start in chunk -> fill until end
                    vad_oh[ch, v_start_frame:] = 1.0
            elif start <= e <= end:  # only end in chunk
                rel_end = e - start
                v_end_frame = round(rel_end / frame_size)
                vad_oh[ch, : v_end_frame + 1] = 1.0
            elif s > end:
                break

    # current speaker is always channel 0
    if speaker == 1:
        vad_oh = torch.stack((vad_oh[1], vad_oh[0]))

    return vad_oh


def get_next_speaker(vad, ds):
    """Doing `get_next_speaker` in reverse"""
    # Reverse Vad
    vad_reversed = vad.flip(dims=(1,))
    ds_reversed = ds.flip(dims=(1,))
    # get "last speaker"
    next_speaker = get_last_speaker(vad_reversed, ds_reversed)
    # reverse back
    next_speaker = next_speaker.flip(dims=(1,))
    return next_speaker


def get_hold_shift_onehot(vad):
    ds = vad_to_dialog_vad_states(vad)
    prev_speaker = get_last_speaker(vad, ds)
    next_speaker = get_next_speaker(vad, ds)
    silence_ids = torch.where(vad.sum(-1) == 0)

    hold_one_hot = torch.zeros_like(prev_speaker)
    shift_one_hot = torch.zeros_like(prev_speaker)

    hold = prev_speaker[silence_ids] == next_speaker[silence_ids]
    hold_one_hot[silence_ids] = hold.long()
    shift_one_hot[silence_ids] = torch.logical_not(hold).long()
    return hold_one_hot, shift_one_hot


# vad context history
def get_vad_condensed_history(vad, t, speaker, bin_end_times=[60, 30, 15, 5, 0]):
    """
    get the vad-condensed-history over the history of the dialog.

    the amount of active seconds are calculated for each speaker in the segments defined by `bin_end_times`
    (starting from 0).
    The idea is to represent the past further away from the current moment in time more granularly.

    for example:
        bin_end_times=[60, 30, 10, 5, 0] extracts activity for each speaker in the intervals:

            [-inf, t-60]
            [t-60, t-30]
            [t-30, t-10]
            [t-10, t-5]
            [t-50, t]

        The final representation is then the ratio of activity for the
        relevant `speaker` over the total activity, for each bin. if there
        is no activity, that is the segments span before the dialog started
        or (unusually) both are silent, then we set the ratio to 0.5, to
        indicate equal participation.

    Argument:
        - vad:      list: [[(0, 3), (4, 6), ...], [...]] list of list of channel start and end time
    """
    n_bins = len(bin_end_times)
    T = t - torch.tensor(bin_end_times)
    bin_times = [0] + T.tolist()

    bins = torch.zeros(2, n_bins)
    for ch, ch_vad in enumerate(vad):  # iterate over each channel
        s = bin_times[0]
        for i, e in enumerate(bin_times[1:]):  # iterate over bin segments
            if e < 0:  # skip if before dialog start
                s = e  # update
                continue
            for vs, ve in ch_vad:  # iterate over channel VAD
                if vs >= s:  # start inside bin time
                    if vs < e and ve <= e:  # both vad_start/end occurs in segment
                        bins[ch][i] += ve - vs
                    elif vs < e:  # only start occurs in segment
                        bins[ch][i] += e - vs
                elif (
                    vs > e
                ):  # all starts occus after bin-end -> no need to process further
                    break
                else:  # vs is before segment
                    if s <= ve <= e:  # ending occurs in segment
                        bins[ch][i] += ve - s
            # update bin start
            s = e
    # Avoid nan -> for loop
    # get the ratio of the relevant speaker
    # if there is no information (bins are before dialog start) we use an equal prior (=.5)
    ratios = torch.zeros(n_bins)
    for b in range(n_bins):
        binsum = bins[:, b].sum()
        if binsum > 0:
            ratios[b] = bins[speaker, b] / binsum
        else:
            ratios[b] = 0.5  # equal prior for segments with no one speaking
    return ratios


@torch.no_grad()
def get_activity_history(vad_frames, bin_end_frames, channel_last=True):
    """

    Uses convolutions to sum the activity over each segment of interest.

    The kernel size is set to be the number of frames of any particular segment i.e.

    ---------------------------------------------------


    ```
    ... h0       | h1 | h2 | h3 | h4 +
    distant past |    |    |    |    +
    -inf -> -t0  |    |    |    |    +

    ```

    ---------------------------------------------------

    Arguments:
        vad_frames:         torch.tensor: (Channels, N_Frames) or (N_Frames, Channels)
        bin_end_frames:     list: boundaries for the activity history windows i.e. [6000, 3000, 1000, 500]
        channel_last:       bool: if true we expect `vad_frames` to be (N_Frames, Channels)

    Returns:
        ratios:             torch.tensor: (Channels, N_frames, bins) or (N_frames, bins, Channels) (dependent on `channel_last`)
        history_bins:       torch.tesnor: same size as ratio but contains the number of active frames, over each segment, for both speakers.
    """

    N = vad_frames.shape[0]
    if channel_last:
        vad_frames = rearrange(vad_frames, "n c -> c n")

    # container for the activity of the defined bins
    hist_bins = []

    # Distance past activity history/ratio
    # The segment from negative infinity to the first bin_end_frames
    if vad_frames.shape[0] > bin_end_frames[0]:
        h0 = vad_frames[:, : -bin_end_frames[0]].cumsum(dim=-1)
        diff_pad = torch.ones(2, bin_end_frames[0]) * -1
        h0 = torch.cat((diff_pad, h0), dim=-1)
    else:
        # there is not enough duration to get any long time information
        # -> set to prior of equal speech
        # negative values for debugging to see where we provide prior
        # (not seen outside of this after r0/r1 further down)
        h0 = torch.ones(2, N) * -1
    hist_bins.append(h0)

    # Activity of segments defined by the the `bin_end_frames`

    # If 0 is not included in the window (i.e. the current frame)
    # we append it for consistency in loop below
    if bin_end_frames[-1] != 0:
        bin_end_frames = bin_end_frames + [0]

    # Loop over each segment window, construct conv1d (summation: all weights are 1.)
    # Omit end-frames which are not used for the current bin
    # concatenate activity sum with pad (= -1) at the start where the bin values are
    # not defined.
    for start, end in zip(bin_end_frames[:-1], bin_end_frames[1:]):
        ks = start - end
        if end > 0:
            vf = vad_frames[:, :-end]
        else:
            vf = vad_frames
        if vf.shape[1] > 0:
            filters = torch.ones((1, 1, ks), dtype=torch.float)
            vf = F.pad(vf, [ks - 1, 0]).unsqueeze(1)  # add channel dim
            o = F.conv1d(vf, weight=filters).squeeze(1)  # remove channel dim
            if end > 0:
                # print('diffpad: ', end)
                diff_pad = torch.ones(2, end) * -1
                o = torch.cat((diff_pad, o), dim=-1)
        else:
            # there is not enough duration to get any long time information
            # -> set to prior of equal speech
            # negative values for debugging to see where we provide prior
            # (not seen outside of this after r0/r1 further down)
            o = torch.ones(2, N) * -1
        hist_bins.append(o)

    # stack together -> (2, N, len(bin_end_frames) + 1) default: (2, N, 5)
    hist_bins = torch.stack(hist_bins, dim=-1)

    # find the ratios for each speaker
    r0 = hist_bins[0] / hist_bins.sum(dim=0)
    r1 = hist_bins[1] / hist_bins.sum(dim=0)

    # segments where both speakers are silent (i.e. [0, 0] activation)
    # are not defined (i.e. hist_bins / hist_bins.sum = 0 / 0 ).
    # Where both speakers are silent they have equal amount of
    nan_inds = torch.where(r0.isnan())
    r0[nan_inds] = 0.5
    r1[nan_inds] = 0.5

    # Consistent input/output with `channel_last` VAD
    if channel_last:
        ratio = torch.stack((r0, r1), dim=-1)
    else:
        ratio = torch.stack((r0, r1))
    return ratio, hist_bins
