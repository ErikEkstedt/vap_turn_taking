# VAP: Voice Activity Projection


Voice Activity Projection module used in the paper [Voice Activity Projection: Self-supervised Learning of Turn-taking Events]().

* VAP-head
  - An NN 'layer' which extracts VAP-labels (discrete, independent, comparative), projection-windows to states, define zero-shot probabilities.
* Events
  - Automatically extract turn-taking events given Voice Activity (e.g. tensor: `(B, N_FRAMES, 2)`) for two speakers
* Metrics
  - [Torchmetrics](https://torchmetrics.readthedocs.io/en/latest/)


## Installation

Install `vap_turn_taking`

* preferably using an environment [miniconda](https://docs.conda.io/en/latest/miniconda.html)
* Including a working installation of [pytorch](https://pytorch.org/)
* [Optional] (for videos) Install FFMPEG: `conda install -c conda-forge ffmpeg`
* Install dependencies: `pip install -r requirements.txt`
* Install package: `pip install -e . `


## VAP
See section 2 of the [paper]().

The Voice Acticity Projection module extract model ('discrete', 'independent',
'comparative') VA-labels and given voice activity and model logits-outputs,
extracts turn-taking ("zero-shot") probabilities.

```python
from vap_turn_taking.config.example_data import example
from vap_turn_taking import VAP


vapper = VAP(type="discrete")

# example of voice activity for 2 speakers
va = example['va']  # Voice Activity (Batch, N_Frames, 2)


# Extract labels: Voice Acticity Projection windows
#   Discrete:       (B, N_frames), class indices
#   Independent:    (B, N_frames, 2, N_bins), binary vap_bins
#   Comaparative:   (B, N_frames), float scalar
y = vapper.extract_label(va)

# Associated logits (discrete/independent/comparative)
logits = model(INPUTS)  # same shape as the labels


# Get "zero-shot" probabilites
turn_taking_probs = vapper(logits, va)  # keys: "p", "p_bc"
# turn_taking_probs['p'], (B, N_frames, 2) -> probability of next speaker
# turn_taking_probs['p_bc'], (B, N_frames, 2) -> probability of backchannel prediction
```


## Events

See section 3 of the [paper]().

The module which extract events from a Voice Activity representation used to
calculate scores over particular frames of interest.

```python
from vap_turn_taking.config.example_data import example, event_conf
from vap_turn_taking import TurnTakingEvents


# example of voice activity for 2 speakers
va = example['va']  # Voice Activity (Batch, N_Frames, 2)


# Class to extract turn-taking events
eventer = TurnTakingEvents(
    hs_kwargs=event_conf["hs"],
    bc_kwargs=event_conf["bc"],
    metric_kwargs=event_conf["metric"],
    frame_hz=100,
)

# extract events from binary voice activity features
events = eventer(va, max_frame=None)

# all events are binary representations of size (B, N_frames, 2)
# where 1 indicates an event relevant frame.
# events.keys(): [
#   'shift', 
#   'hold', 
#   'short', 
#   'long', 
#   'predict_shift_pos', 
#   'predict_shift_neg', 
#   'predict_bc_pos', 
#   'predict_bc_neg'
# ]
```

Where the `event_kwargs` can be

```python
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
```


## Metrics

See section 3 of the [paper]().

Calculates metrics during training/evaluation given the `turn_taking_probs`
from the `VAP`+model-output and the events from `TurnTakingEvents`. Built using [torchmetrics](https://torchmetrics.readthedocs.io/en/latest/).

```python
from vap_turn_taking import TurnTakingMetrics
from vap_turn_taking.config.example_data import example, event_conf


va = example['va']  # Voice Activity (Batch, N_Frames, 2)


metric = TurnTakingMetrics(
    hs_kwargs=event_conf["hs"],
    bc_kwargs=event_conf["bc"],
    metric_kwargs=event_conf["metric"],
    bc_pred_pr_curve=True,
    shift_pred_pr_curve=True,
    long_short_pr_curve=True,
    frame_hz=100,
)

# Forward pass through a model, extract events, extract turn-taking probabilites
logits = model(INPUTS)
events = eventer(va, max_frame=None)
turn_taking_probs = vapper(logits, va)  # keys: "p", "p_bc"

# Update metrics
metric.update(
    p=turn_taking_probs["p"],
    bc_pred_probs=turn_taking_probs.get("bc_prediction", None),
    events=events,
)

# Compute: finalize/aggregates the scores (usually used after epoch is finished)
result = metric.compute()

# Resets the metrics (usually used before starting a new epoch)
result = metric.reset()
```

## Citation

```latex
TBA
```
