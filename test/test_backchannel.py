import pytest
import torch

from vap_turn_taking.backchannel import extract_backchannel_prediction_probs_independent


@pytest.mark.backchannel
def test_independent_bc_prediction():

    bc, lab = [], []
    # A provides backchannel
    bc.append(torch.tensor([[0.2, 0.6, 0.3, 0.1], [0.5] * 4]))
    lab.append(1)
    bc.append(torch.tensor([[0.2, 0.6, 0.7, 0.1], [0.5] * 4]))
    lab.append(1)
    bc.append(torch.tensor([[0.55, 0.1, 0.1, 0.1], [0.5] * 4]))
    lab.append(1)
    # A does not provide backchannel
    bc.append(torch.tensor([[0.55, 0.4, 0.4, 0.6], [0.5] * 4]))  # between false
    lab.append(0)
    bc.append(torch.tensor([[0.3, 0.3, 0.3, 0.4], [0.5] * 4]))  # wihin false
    lab.append(0)
    bc = torch.stack(bc)
    lab = torch.tensor(lab)

    a_bc_pred = extract_backchannel_prediction_probs_independent(bc)
    b_bc_pred = extract_backchannel_prediction_probs_independent(bc.flip(1))

    a_err = ((a_bc_pred[..., 0] > 0) != lab).sum()
    b_err = ((b_bc_pred[..., 1] > 0) != lab).sum()

    assert a_err == 0, "A bc error"
    assert b_err == 0, "B bc error"
