import pytest
from vap_turn_taking import HoldShift
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
