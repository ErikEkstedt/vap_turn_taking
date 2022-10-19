import pytest
import torch
from vap_turn_taking.hold_shifts import HoldShift, HoldShiftNew
from vap_turn_taking.config.example_data import example, event_conf_frames


@pytest.mark.hold_shift
def test_hold_shifts():

    hs_kwargs = event_conf_frames["hs"]
    HS = HoldShift(**hs_kwargs)
    tt = HS(example["va"])

    hdiff = (tt["hold"] != example["hold"]).sum()
    sdiff = (tt["shift"] != example["shift"]).sum()

    assert hdiff == 0, f"Backchannel hold diff {hdiff} != 0"
    assert sdiff == 0, f"Backchannel shift diff {sdiff} != 0"


@pytest.mark.hold_shift
def test_hold_shifts_new():

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )

    HS = HoldShiftNew(
        pre_cond_time=1.0,
        post_cond_time=1.0,
        prediction_region_time=0.2,
        prediction_region_on_active=True,
        long_onset_condition_time=1.0,
        long_onset_region_time=0.2,
        min_silence_time=0.2,
        min_context_time=3.0,
        max_time=10.0,
        frame_hz=50,
    )
    hs = HS(vad)

    LABEL_SHIFT = [1, 0, 1]
    LABEL_HOLD = [1, 2, 1]

    s_lens = [len(s) for s in hs["shift"]]
    h_lens = [len(h) for h in hs["hold"]]

    assert s_lens == LABEL_SHIFT, "Error shift"
    assert h_lens == LABEL_HOLD, "Error hold"
