import torch

from vap_turn_taking.utils import time_to_frames

# TODO:Decide on config format/class/yaml/json


# Configs for Events
metric_kwargs = dict(
    pad=0,  # int, pad on silence (shift/hold) onset used for evaluating\
    dur=0.2,  # int, duration off silence (shift/hold) used for evaluating\
    pre_label_dur=0.4,  # int, frames prior to Shift-silence for prediction on-active shift
    onset_dur=0.2,
    min_context=3,
)
hs_kwargs = dict(
    post_onset_shift=1,
    pre_offset_shift=1,
    post_onset_hold=1,
    pre_offset_hold=1,
    non_shift_horizon=2,
    metric_pad=metric_kwargs["pad"],
    metric_dur=metric_kwargs["dur"],
    metric_pre_label_dur=metric_kwargs["pre_label_dur"],
    metric_onset_dur=metric_kwargs["onset_dur"],
)
bc_kwargs = dict(
    max_duration_frames=1,
    pre_silence_frames=1,
    post_silence_frames=1,
    min_duration_frames=metric_kwargs["onset_dur"],
    metric_dur_frames=metric_kwargs["onset_dur"],
    metric_pre_label_dur=metric_kwargs["pre_label_dur"],
)
event_conf = {"hs": hs_kwargs, "bc": bc_kwargs, "metric": metric_kwargs}
######################################################################################
frame_hz = 100
event_conf_frames = {}
for k, v in event_conf.items():
    event_conf_frames[k] = {}
    for kk, vv in v.items():
        if kk != "non_shift_majority_ratio":
            event_conf_frames[k][kk] = time_to_frames(vv, frame_hz)

######################################################################################
a_bc = (585, 660)
as_onset = (840, 950)
a_post_hold = (1000, 1100)
start = 1110
b_bc = (start, start + 80)
bs_onset = (350, 460)
A_segments = [(0, 300), a_bc, as_onset, a_post_hold, (1200, 1360)]
B_segments = [bs_onset, (480, 590), (670, 800), b_bc, (1350, 1590)]
max_frame = max(A_segments[-1][-1], B_segments[-1][-1])
va = torch.zeros((max_frame, 2), dtype=torch.float)
for start, end in A_segments:
    va[start:end, 0] = 1.0
for start, end in B_segments:
    va[start:end, 1] = 1.0
# Labels
onset = event_conf_frames["metric"]["onset_dur"]
dur = event_conf_frames["metric"]["dur"]
# BC
bc = torch.zeros_like(va)
bc[a_bc[0] : a_bc[0] + onset, 0] = 1.0
bc[b_bc[0] : b_bc[0] + onset, 1] = 1.0
# S/H
s = torch.zeros_like(va)
s[300 : 300 + dur, 1] = 1.0
s[800 : 800 + dur, 0] = 1.0
# HOLD
h = torch.zeros_like(va)
h[as_onset[-1] : as_onset[-1] + dur, 0] = 1.0
h[bs_onset[-1] : bs_onset[-1] + dur, 1] = 1.0
# Long
long = torch.zeros_like(va)
long[350 : 350 + onset, 1] = 1
long[840 : 840 + onset, 0] = 1
# Long
short = torch.zeros_like(va)
short[a_bc[0] : a_bc[0] + onset, 0] = 1
short[b_bc[0] : b_bc[0] + onset, 1] = 1
# unsqueeze
example = {
    "va": va.unsqueeze(0),
    "hold": h.unsqueeze(0),
    "shift": s.unsqueeze(0),
    "backchannel": bc.unsqueeze(0),
    "short": short.unsqueeze(0),
    "long": long.unsqueeze(0),
}
