import pytest

import torch
import vap_turn_taking.functional as VF
from vap_turn_taking.utils import vad_list_to_onehot


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


@pytest.mark.functional
def test_backchannel(data):
    pre_cond_frames = 50
    post_cond_frames = 50
    min_context_frames = 150
    max_bc_frames = 50
    max_frame = 500
    # vad = data["bc"]["vad"][0]  # (n_frames, 2)

    import matplotlib.pyplot as plt
    from vap_turn_taking.plot_utils import plot_vad_oh

    vad_lists = [
        [[[2, 4], [7, 9]], [[4.6, 5.5]]],
        [[[2, 4]], [[4.6, 5.5], [7, 9]]],  # not bc with filleed vad
        [[[2, 4]], [[4.6, 5.5]]],  # no activity after (can still be bc)
        [[[2, 4], [6.8, 7.1]], [[4.6, 5.5], [7, 9]]],  # not bc with filleed vad
    ]
    n_bc = [1, 0, 1, 2]

    for vad_list, N in zip(vad_lists, n_bc):
        vad = vad_list_to_onehot(
            vad_list, hop_time=0.02, duration=11.99, channel_last=True
        )
        backchannels = VF.backchannel_regions(
            vad,
            pre_cond_frames=pre_cond_frames,
            post_cond_frames=post_cond_frames,
            min_context_frames=min_context_frames,
            max_bc_frames=max_bc_frames,
            max_frame=max_frame,
        )
        # ds = VF.get_dialog_states(vad)
        # filled_vad = VF.fill_pauses(vad)
        # fig, [ax, ax1] = plt.subplots(2, 1, figsize=(9, 6))
        # _ = plot_vad_oh(vad, ax=ax)
        # _ = plot_vad_oh(filled_vad, ax=ax1)
        # ax.axvline(min_context_frames, linewidth=4, color="k")
        # ax.axvline(max_frame, linewidth=4, color="k")
        # for bc_start, bc_end, speaker in backchannels:
        #     ymin = 0
        #     ymax = 1
        #     if speaker == 1:
        #         ymin = -1
        #         ymax = 0
        #     ax.vlines(bc_start, ymin=ymin, ymax=ymax, linewidth=4, color="g")
        #     ax.vlines(bc_end, ymin=ymin, ymax=ymax, linewidth=4, color="r")
        # plt.show()
        assert len(backchannels) == N, "Wrong number of backchannels found"
