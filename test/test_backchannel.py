import pytest

import torch
from vap_turn_taking.backchannel import Backchannel, BackchannelNew
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


@pytest.mark.backchannel()
def test_backchannel_new():

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )

    BC = BackchannelNew(
        pre_cond_time=1.0,
        post_cond_time=1.0,
        prediction_region_time=0.2,
        max_bc_duration=1.0,
        min_context_time=3.0,
        max_time=10.0,
        frame_hz=50,
    )

    bcs = BC(vad)

    # Number of backchannels in vad sample
    LABEL_BC = [0, 0, 1]
    b_lens = [len(b) for b in bcs["backchannel"]]
    assert b_lens == LABEL_BC, "Error number of backchannels"
