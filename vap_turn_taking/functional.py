import torch
from typing import Dict, Tuple, List, Optional


# Templates
TRIAD_SHIFT: torch.Tensor = torch.tensor([[3, 1, 0], [0, 1, 3]])  # on Silence
TRIAD_SHIFT_OVERLAP: torch.Tensor = torch.tensor([[3, 2, 0], [0, 2, 3]])
TRIAD_HOLD: torch.Tensor = torch.tensor([[0, 1, 0], [3, 1, 3]])  # on silence
TRIAD_BC: torch.Tensor = torch.tensor([0, 1, 0])

# Dialog states meaning
STATE_ONLY_A: int = 0
STATE_ONLY_B: int = 3
STATE_SILENCE: int = 1
STATE_BOTH: int = 2


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


def fill_pauses(
    vad: torch.Tensor,
    ds: torch.Tensor,
    islands: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    assert vad.ndim == 2, "fill_pauses require ds=(n_frames, 2)"
    assert ds.ndim == 1, "fill_pauses require ds=(n_frames,)"

    filled_vad = vad.clone()

    if islands is None:
        s, d, v = find_island_idx_len(ds)
    else:
        s, d, v = islands

    # less than three entries means that there are no pauses
    # requires at least: activity-from-speaker  ->  Silence   --> activity-from-speaker
    if len(v) < 3:
        return vad

    triads = v.unfold(0, size=3, step=1)
    next_speaker, steps = torch.where((triads == TRIAD_HOLD.unsqueeze(1)).sum(-1) == 3)
    for ns, pre in zip(next_speaker, steps):
        cur = pre + 1
        # Fill the matching template
        filled_vad[s[cur] : s[cur] + d[cur], ns] = 1.0
    return filled_vad


def hold_shift_regions_simple(
    ds: torch.Tensor,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Extract Holds/Shifts dirctly from ds=dialog_states without any type of condtions.

    That is do not check what happes before/after the mutual silence so you get
    holds/shifts that are solely defined on what frame the active frame before and after the silence
    belongs to.
    """
    assert ds.ndim == 1, f"expects ds of shape (n_frames, ) but got {ds.shape}."

    def _get_regions(
        triads: torch.Tensor, indices: torch.Tensor, triad_label: torch.Tensor
    ) -> List[Tuple[int, int]]:
        """
        All shift triads e.g. [3, 1, 0] centers on the silence segment
        If we find a triad match at step 's' then the actual SILENCE segment
        STARTS on the next step -> add 1
        ENDS   on the two next step -> add 2
        """
        region = []
        _, steps = torch.where((triads == triad_label.unsqueeze(1)).sum(-1) == 3)
        for step in steps:
            silence_start = indices[step + 1].item()  # start of
            silence_end = indices[step + 2].item()  # add
            region.append((silence_start, silence_end))
        return region

    indices, dur, states = find_island_idx_len(ds)

    # If we have less than 3 unique dialog states
    # then we have no valid transitions
    if len(states) < 3:
        return [], [], []

    triads = states.unfold(0, size=3, step=1)

    shifts = _get_regions(triads, indices, TRIAD_SHIFT)
    shift_overlap = _get_regions(triads, indices, TRIAD_SHIFT_OVERLAP)
    holds = _get_regions(triads, indices, TRIAD_HOLD)

    return shifts, shift_overlap, holds


def get_hs_regions(
    triads: torch.Tensor,
    filled_vad: torch.Tensor,
    triad_label: torch.Tensor,
    start_of: torch.Tensor,
    duration_of: torch.Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    prediction_region_frames: int,
    prediction_region_on_active: bool,
    min_silence_frames: int,
    min_context_frames: int,
    max_frame: int,
) -> Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]:
    """
    get regions defined by `triad_label`
    """

    region = []
    prediction_region = []

    # check if label is hold or shift
    # if the same speaker continues after silence -> hold
    hold_cond = triad_label[0, 0] == triad_label[0, -1]
    next_speakers, steps = torch.where(
        (triads == triad_label.unsqueeze(1)).sum(-1) == 3
    )
    # No matches -> return
    if len(next_speakers) == 0:
        return [], []

    for last_onset, next_speaker in zip(steps, next_speakers):
        not_next_speaker = 0 if next_speaker == 1 else 1
        prev_speaker = next_speaker if hold_cond else not_next_speaker
        not_prev_speaker = 0 if prev_speaker == 1 else 1
        # All shift triads e.g. [3, 1, 0] centers on the silence segment
        # If we find a triad match at step 's' then the actual SILENCE segment
        # STARTS:           on the next step -> add 1
        # ENDS/next-onset:  on the two next step -> add 2
        silence = last_onset + 1
        next_onset = last_onset + 2
        ################################################
        # MINIMAL CONTEXT CONDITION
        ################################################
        if start_of[silence] < min_context_frames:
            continue
        ################################################
        # MAXIMAL FRAME CONDITION
        ################################################
        if start_of[silence] >= max_frame:
            continue
        ################################################
        # MINIMAL SILENCE CONDITION
        ################################################
        # Check silence duration
        if duration_of[silence] < min_silence_frames:
            continue
        ################################################
        # PRE CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
        ################################################
        # Check `pre_cond_frames` before start of silence
        # to make sure only a single speaker was active
        sil_start = start_of[silence]
        pre_start = sil_start - pre_cond_frames
        pre_start = pre_start if pre_start > 0 else 0
        correct_is_active = (
            filled_vad[pre_start:sil_start, prev_speaker].sum() == pre_cond_frames
        )
        if not correct_is_active:
            continue
        other_is_silent = filled_vad[pre_start:sil_start, not_prev_speaker].sum() == 0
        if not other_is_silent:
            continue
        ################################################
        # POST CONDITION: ONLY A SINGLE PREVIOUS SPEAKER
        ################################################
        # Check `post_cond_frames` after start of onset
        # to make sure only a single speaker is to be active
        onset_start = start_of[next_onset]
        onset_region_end = onset_start + post_cond_frames
        correct_is_active = (
            filled_vad[onset_start:onset_region_end, next_speaker].sum()
            == post_cond_frames
        )
        if not correct_is_active:
            continue
        other_is_silent = (
            filled_vad[onset_start:onset_region_end, not_next_speaker].sum() == 0
        )
        if not other_is_silent:
            continue
        ################################################
        # ALL CONDITIONS MET
        ################################################
        region.append((sil_start.item(), onset_start.item(), next_speaker))

        ################################################
        # PREDICTION REGION CONDITION
        ################################################
        # The prediction region is defined at the end of the previous
        # activity, not inside the silences.

        # IF PREDICTION_REGION_ON_ACTIVE = FALSE
        # We don't care about the previous activity but only take
        # `prediction_region_frames` prior to the relevant hold/shift silence.
        # e.g. if prediction_region_frames=100 and the last segment prior to the
        # relevant hold/shift silence was 70 frames the prediction region would include
        # < 30 frames of silence (a pause or a shift (could be a quick back and forth limited by the condition variables...))
        if prediction_region_on_active:
            # We make sure that the last VAD segments
            # of the last speaker is longer than
            # `prediction_region_frames`
            if duration_of[last_onset] < prediction_region_frames:
                continue

        # that if the last activity
        prediction_start = sil_start.item() - prediction_region_frames

        ################################################
        # MINIMAL CONTEXT CONDITION (PREDICTION)
        ################################################
        if prediction_start < min_context_frames:
            continue

        prediction_region.append((prediction_start, sil_start.item(), next_speaker))

    return region, prediction_region


def hold_shift_regions(
    vad: torch.Tensor,
    ds: torch.Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    prediction_region_frames: int,
    prediction_region_on_active: bool,
    min_silence_frames: int,
    min_context_frames: int,
    max_frame: int,
) -> Dict[str, List[Tuple[int, int, int]]]:
    assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."
    assert ds.ndim == 1, f"expects ds of shape (n_frames, ) but got {ds.shape}."

    start_of, duration_of, states = find_island_idx_len(ds)
    filled_vad = fill_pauses(vad, ds, islands=(start_of, duration_of, states))

    # If we have less than 3 unique dialog states
    # then we have no valid transitions
    if len(states) < 3:
        return {"shift": [], "hold": [], "pred_shift": [], "pred_hold": []}

    triads = states.unfold(0, size=3, step=1)

    shifts, pred_shifts = get_hs_regions(
        triads=triads,
        filled_vad=filled_vad,
        triad_label=TRIAD_SHIFT,
        start_of=start_of,
        duration_of=duration_of,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        prediction_region_frames=prediction_region_frames,
        prediction_region_on_active=prediction_region_on_active,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )

    holds, pred_holds = get_hs_regions(
        triads=triads,
        filled_vad=filled_vad,
        triad_label=TRIAD_HOLD,
        start_of=start_of,
        duration_of=duration_of,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        prediction_region_frames=prediction_region_frames,
        prediction_region_on_active=prediction_region_on_active,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )
    return {
        "shift": shifts,
        "hold": holds,
        "pred_shift": pred_shifts,
        "pred_hold": pred_holds,
    }


def backchannel_regions(
    vad: torch.Tensor,
    pre_cond_frames: int,
    post_cond_frames: int,
    min_context_frames: int,
    max_bc_frames: int,
    max_frame: int,
):
    assert vad.ndim == 2, f"expects vad of shape (n_frames, 2) but got {vad.shape}."

    ds = get_dialog_states(vad)
    filled_vad = fill_pauses(vad, ds)

    backchannels = []
    for speaker in [0, 1]:
        start_of, duration_of, states = find_island_idx_len(filled_vad[..., speaker])
        if len(states) < 3:
            continue
        triads = states.unfold(0, size=3, step=1)
        steps = torch.where((triads == TRIAD_BC.unsqueeze(0)).sum(-1) == 3)[0]
        if len(steps) == 0:
            continue
        for pre_silence in steps:
            bc = pre_silence + 1
            post_silence = pre_silence + 2
            ################################################
            # MINIMAL CONTEXT CONDITION
            ################################################
            if start_of[bc] < min_context_frames:
                # print("Minimal context")
                continue
            ################################################
            # MAXIMAL FRAME CONDITION
            ################################################
            if start_of[bc] >= max_frame:
                # print("Max frame")
                continue
            ################################################
            # MINIMAL DURATION CONDITION
            ################################################
            # Check bc duration
            if duration_of[bc] > max_bc_frames:
                # print("Too Long")
                continue
            ################################################
            # PRE CONDITION: No previous activity from bc-speaker
            ################################################
            if duration_of[pre_silence] < pre_cond_frames:
                # print('not enough silence PRIOR to "bc"')
                continue
            ################################################
            # POST CONDITION: No post activity from bc-speaker
            ################################################
            if duration_of[post_silence] < post_cond_frames:
                # print('not enough silence POST to "bc"')
                continue
            ################################################
            # PRE OTHER CONDITION
            ################################################
            # Is the other speakr active before this segment?
            backchannels.append((start_of[bc], start_of[post_silence], speaker))

    return backchannels
