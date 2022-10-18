import pytest

import torch
import vap_turn_taking.functional as VF


@pytest.fixture
def data():
    data = torch.load("example/vap_data.pt")
    return data


@pytest.mark.functional
def test_dialog_state(data):
    vad = data["shift"]["vad"]  # (1, 600, 2)
    B, N_FRAMES, _ = vad.shape
    ds = VF.get_dialog_states(vad)
    assert ds.shape == (
        B,
        N_FRAMES,
    ), f"get_dialog_states expected shape {(B, N_FRAMES)} got {(tuple(ds.shape))}."


@pytest.mark.functional
def test_find_island_idx_len(data):

    vad = data["shift"]["vad"]  # (1, 600, 2)
    B, N_FRAMES, _ = vad.shape

    # get dialog states
    ds = VF.get_dialog_states(vad)
    assert ds.shape == (
        B,
        N_FRAMES,
    ), f"get_dialog_states expected shape {(B, N_FRAMES)} got {(tuple(ds.shape))}."

    idx, dur, values = VF.find_island_idx_len(ds[0])
    assert (
        len(idx) == len(dur) == len(values)
    ), f"lengths does not match: idx != dur != values"
    assert len(idx) == 14, f"IDX length does not match, {len(idx)} != 14"


@pytest.mark.functional
def test_fill_pauses(data):
    vad = data["shift"]["vad"]  # (1, 600, 2)
    vad = vad[0]  # (600, 2)

    ds = VF.get_dialog_states(vad)
    filled_vad = VF.fill_pauses(vad, ds)
    # import matplotlib.pyplot as plt
    # from vap_turn_taking.plot_utils import plot_vad_oh
    # fig, ax = plt.subplots(2, 1, figsize=(9,6))
    # _ = plot_vad_oh(vad, ax=ax[0])
    # _ = plot_vad_oh(filled_vad, ax=ax[1])
    # plt.show()
    assert (
        vad.shape == filled_vad.shape
    ), f"shaps don't match. vad: {vad.shape} != filled_vad: {filled_vad.shape}"


@pytest.mark.functional
def test_shift_hold_simple(data):
    vad = data["shift"]["vad"]  # (1, 600, 2)
    vad = vad[0]  # (600, 2)
    ds = VF.get_dialog_states(vad)

    shifts, shift_overlap, holds = VF.hold_shift_regions_simple(ds)
    assert len(shifts) == 2, f"Number of shifts are not correct {len(shifts)} != 2"
    assert len(holds) == 4, f"Number of holds are not correct {len(holds)} != 4"
    assert len(shift_overlap) == 0, f"Number of holds are not correct {len(holds)} != 0"

    vad = data["bc"]["vad"]  # (1, 600, 2)
    vad = vad[0]  # (600, 2)
    ds = VF.get_dialog_states(vad)

    shifts, shift_overlap, holds = VF.hold_shift_regions_simple(ds)
    # import matplotlib.pyplot as plt
    # from vap_turn_taking.plot_utils import plot_vad_oh
    # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # _ = plot_vad_oh(vad, ax=ax)
    # for start, end in shifts:
    #     ax.axvline(start, linewidth=2, color="g")
    #     ax.axvline(end, linewidth=2, color="r")
    # for start, end in holds:
    #     ax.axvline(start, linewidth=2, linestyle="dashed", color="g")
    #     ax.axvline(end, linewidth=2, linestyle="dashed", color="r")
    # plt.pause(0.1)
    assert len(shifts) == 1, f"Number of shifts are not correct {len(shifts)} != 1"
    assert len(holds) == 5, f"Number of holds are not correct {len(holds)} != 4"
    assert len(shift_overlap) == 0, f"Number of holds are not correct {len(holds)} != 0"


@pytest.mark.functional
def test_shift_hold(data):
    """
    Test shift/hold extraction on a single sample
    """
    # @ 50hz
    pre_cond_frames = 50  # 1 second
    post_cond_frames = 50  # 1 second
    min_silence_frames = 10  # 0.2 second
    min_context_frames = 150  # 3 seconds
    max_frame = 500  # 10 seconds

    # Input
    vad = data["shift"]["vad"][0]  # (600,2)
    ds = VF.get_dialog_states(vad)
    shifts, holds = VF.hold_shift_regions(
        vad=vad,
        ds=ds,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )
    assert len(shifts) == 1, f"Number of shifts are incorrect {len(shifts)} != 1"
    assert len(holds) == 1, f"Number of holds are incorrect {len(holds)} != 1"

    # Input
    vad = data["only_hold"]["vad"][0]  # (600,2)
    ds = VF.get_dialog_states(vad)
    shifts, holds = VF.hold_shift_regions(
        vad=vad,
        ds=ds,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )
    assert len(shifts) == 0, f"Number of shifts are incorrect {len(shifts)} != 0"
    assert len(holds) == 2, f"Number of holds are incorrect {len(holds)} != 2"

    # Input
    vad = data["bc"]["vad"][0]  # (600,2)
    ds = VF.get_dialog_states(vad)
    shifts, holds = VF.hold_shift_regions(
        vad=vad,
        ds=ds,
        pre_cond_frames=pre_cond_frames,
        post_cond_frames=post_cond_frames,
        min_silence_frames=min_silence_frames,
        min_context_frames=min_context_frames,
        max_frame=max_frame,
    )
    # import matplotlib.pyplot as plt
    # from vap_turn_taking.plot_utils import plot_vad_oh
    # fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # _ = plot_vad_oh(vad, ax=ax)
    # ax.axvline(min_context_frames, linewidth=4, color="k")
    # ax.axvline(max_frame, linewidth=4, color="k")
    # for start, end in shifts:
    #     ax.axvline(start, linewidth=2, color="g")
    #     ax.axvline(end, linewidth=2, color="r")
    # for start, end in holds:
    #     ax.axvline(start, linewidth=2, linestyle="dashed", color="g")
    #     ax.axvline(end, linewidth=2, linestyle="dashed", color="r")
    # plt.pause(0.1)
    assert len(shifts) == 1, f"Number of shifts are incorrect {len(shifts)} != 1"
    assert len(holds) == 1, f"Number of holds are incorrect {len(holds)} != 1"


@pytest.mark.functional
def test_shift_hold_batch(data):
    """
    Test shift/hold extraction on batched sample
    """

    # @ 50hz
    pre_cond_frames = 50  # 1 second
    post_cond_frames = 50  # 1 second
    min_silence_frames = 10  # 0.2 second
    min_context_frames = 150  # 3 seconds
    max_frame = 500  # 10 seconds

    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    batch_size = vad.shape[0]

    shifts, holds = [], []
    for b in range(batch_size):
        ds = VF.get_dialog_states(vad[b])
        tmp_shifts, tmp_holds = VF.hold_shift_regions(
            vad=vad[b],
            ds=ds,
            pre_cond_frames=pre_cond_frames,
            post_cond_frames=post_cond_frames,
            min_silence_frames=min_silence_frames,
            min_context_frames=min_context_frames,
            max_frame=max_frame,
        )
        shifts.append(tmp_shifts)
        holds.append(tmp_holds)
    assert (
        len(shifts) == batch_size
    ), f"not all SHIFT samples are included batch_size is off"
    assert (
        len(holds) == batch_size
    ), f"not all HOLD samples are included batch_size is off"
