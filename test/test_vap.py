import pytest
import torch
from vap_turn_taking import VAP
from vap_turn_taking.vap_new import VAP as VAPNew
from vap_turn_taking.config.example_data import example


@pytest.mark.vap
def test_vap_discrete():
    vapper = VAP(type="discrete")
    va = example["va"]
    y = vapper.extract_label(va)
    assert y.ndim == 2, "Shape error: {y.shape} != (B, N)"


@pytest.mark.vap
def test_vap_independent():
    vapper = VAP(type="independent")
    y = vapper.extract_label(example["va"])
    assert y.ndim == 4, "Shape error: y.ndim != 4. i.e. not (b, n, 2, 4)"
    assert y.shape[-2:] == (2, 4), f"Shape error: {y.shape[-2]} != (..., c, n_bins)"


@pytest.mark.vap
def test_vap_comparative():
    vapper = VAP(type="comparative")
    va = example["va"]
    y = vapper.extract_label(va)
    assert y.ndim == 2, "Shape error: {y.shape} != (B, N)"


@pytest.mark.vap
def test_vap_new():

    data = torch.load("example/vap_data.pt")

    vad = data["shift"]["vad"]
    for objective in ["discrete", "independent", "comparative"]:
        vap_objective = VAPNew(objective=objective)

        if objective == "comparative":
            logits = torch.randn((1, 500))
        elif objective == "independent":
            logits = torch.randn((1, 500, 2, vap_objective.n_bins))
        else:
            logits = torch.rand((1, 600, 256)) / 256

        out = vap_objective(logits, vad)
        labels = vap_objective.extract_labels(vad)
        loss = vap_objective.loss_fn(logits, labels)
