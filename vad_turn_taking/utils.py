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
