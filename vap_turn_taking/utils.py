import torch


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
