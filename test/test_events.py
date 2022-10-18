import pytest

import torch
from vap_turn_taking.events import TurnTakingEvents, TurnTakingEventsNew
from vap_turn_taking.config.example_data import example, event_conf


@pytest.mark.events
def test_events():
    eventer = TurnTakingEvents(
        hs_kwargs=event_conf["hs"],
        bc_kwargs=event_conf["bc"],
        metric_kwargs=event_conf["metric"],
        frame_hz=100,
    )

    va = example["va"]
    events = eventer(va)

    # Shift/Hold
    sdiff = (events["shift"] != example["shift"]).sum()
    assert sdiff == 0, f"SHIFT non-zero diff: {sdiff}"

    hdiff = (events["hold"] != example["hold"]).sum()
    assert hdiff == 0, f"HOLD non-zero diff: {hdiff}"

    long_diff = (events["long"] != example["long"]).sum()
    assert long_diff == 0, f"LONG non-zero diff: {long_diff}"

    short_diff = (events["short"] != example["short"]).sum()
    assert short_diff == 0, f"SHORT non-zero diff: {short_diff}"


@pytest.mark.events
def test_events_new():

    data = torch.load("example/vap_data.pt")
    vad = torch.cat(
        (
            data["shift"]["vad"],
            data["only_hold"]["vad"],
            data["bc"]["vad"],
        )
    )
    vad = torch.cat([vad] * 10)
    eventer = TurnTakingEventsNew()
    events = eventer(vad)

    holds = events["hold"]
    shifts = events["shift"]
    backchannels = events["backchannel"]

    # N events per repeated "triad"
    LABEL_SHIFT = [1, 0, 1]
    LABEL_HOLD = [1, 2, 1]
    LABEL_BC = [0, 0, 1]

    last_i = 0
    for i in range(3, 30, 3):
        h = holds[last_i:i]
        s = shifts[last_i:i]
        b = backchannels[last_i:i]
        h_lens = [len(hh) for hh in h]
        s_lens = [len(ss) for ss in s]
        b_lens = [len(bb) for bb in b]
        assert h_lens == LABEL_HOLD, "Error holds"
        assert s_lens == LABEL_SHIFT, "Error shift"
        assert b_lens == LABEL_BC, "Error holds"
        last_i = i
