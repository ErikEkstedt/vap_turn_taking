import pytest

from vap_turn_taking import TurnTakingEvents
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
