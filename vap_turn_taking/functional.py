import torch

from typing import Tuple


# Templates


def get_dialog_states(vad: torch.Tensor) -> torch.Tensor:
    """Vad to the full state of a 2 person vad dialog
    0: only speaker 0
    1: none
    2: both
    3: only speaker 1
    """
    assert vad.ndim >= 1
    return (2 * vad[..., 1] - vad[..., 0]).long() + 1


def find_island_idx_len(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def fill_pauses(vad: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    assert vad.ndim == 2, "fill_pauses require ds=(n_frames, 2)"
    assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"

    filled_vad = vad.clone()

    s, d, v = find_island_idx_len(ds)

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        continue
    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where((triads == template.unsqueeze(1)).sum(-1) == 3)
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur] : s[cur] + d[cur], ns] = 1.0
    return filled_vad
