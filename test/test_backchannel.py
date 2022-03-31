import pytest
from vap_turn_taking import Backchannel
from vap_turn_taking.config.example_data import event_conf_frames, example


@pytest.mark.backchannel()
def test_backchannel():

    va = example["va"]
    bc_label = example["backchannel"]

    bc_kwargs = event_conf_frames["bc"]
    bcer = Backchannel(**bc_kwargs)
    tt_bc = bcer(va, max_frame=None)

    ndiff = (bc_label != tt_bc["backchannel"]).sum().item()
    assert ndiff == 0, f"Backchannel diff {ndiff} != 0"
