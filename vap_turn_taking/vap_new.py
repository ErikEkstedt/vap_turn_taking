from typing import List
from vap_turn_taking.discrete import DiscreteVAP
from vap_turn_taking.independent import IndependentVAP, ComparativeVAP


class VAP:
    OBJECTIVES = ["discrete", "independent", "comparative"]

    def __new__(
        cls,
        objective: str = "discrete",
        bin_times: List[float] = [0.20, 0.40, 0.60, 0.80],
        frame_hz: int = 50,
        pre_frames: int = 2,
        threshold_ratio: float = 0.5,
    ):

        if objective == "independent":
            return IndependentVAP(
                bin_times=bin_times,
                pre_frames=pre_frames,
                frame_hz=frame_hz,
                threshold_ratio=threshold_ratio,
            )
        elif objective == "comparative":
            return ComparativeVAP(
                bin_times=bin_times, frame_hz=frame_hz, threshold_ratio=threshold_ratio
            )
        else:
            return DiscreteVAP(
                bin_times=bin_times, frame_hz=frame_hz, threshold_ratio=threshold_ratio
            )


def __ind_main():
    import torch

    data = torch.load("example/vap_data.pt")

    vap_objective = VAP(objective="independent")
    print()
    print(vap_objective)
    vad = data["shift"]["vad"]
    logits = torch.randn((1, 500, 2, vap_objective.n_bins))
    out = vap_objective(logits, vad)
    labels = vap_objective.extract_labels(vad)
    loss = vap_objective.loss_fn(logits, labels)
    print("-" * 45)
    print("vad: ", tuple(vad.shape))
    print("logits: ", tuple(logits.shape))
    print("labels: ", tuple(labels.shape))
    print("out['labels']: ", tuple(out["labels"].shape))
    print("out['probs']: ", tuple(out["probs"].shape))
    print("out['p']: ", tuple(out["p"].shape))
    print("out['p_bc']: ", tuple(out["p_bc"].shape))
    print("Loss: ", loss)
    print("-" * 45)
    pass


def __comp_main():
    vap_objective = VAP(objective="comparative")
    print()
    print(vap_objective)
    vad = data["shift"]["vad"]
    logits = torch.randn((1, 500))
    out = vap_objective(logits, vad)
    labels = vap_objective.extract_labels(vad)
    loss = vap_objective.loss_fn(logits, labels)
    print("-" * 45)
    print("vad: ", tuple(vad.shape))
    print("logits: ", tuple(logits.shape))
    print("labels: ", tuple(labels.shape))
    print("out['labels']: ", tuple(out["labels"].shape))
    print("out['probs']: ", tuple(out["probs"].shape))
    print("out['p']: ", tuple(out["p"].shape))
    print("-" * 45)


if __name__ == "__main__":

    from vap_turn_taking.utils import read_json
    import matplotlib.pyplot as plt
    import torch

    data = torch.load("example/vap_data.pt")
    out = read_json("example/test_probs.json")

    vap_objective = VAP()
    print()
    print(vap_objective)

    # vad = data["shift"]["vad"]
    # logits = torch.rand((1, 600, 256)) / 256
    # out = vap_objective(logits, vad)

    p = torch.tensor(out["p"]).unsqueeze(0)
    print("p: ", tuple(p.shape))
    probs = torch.tensor(out["probs"]).unsqueeze(0)
    print("probs: ", tuple(probs.shape))
    vad = torch.tensor(out["vad"]).unsqueeze(0)
    vad = (vad > 0.5).float()
    print("vad: ", tuple(vad.shape))

    fig, ax = plt.subplots(2, 1, figsize=(12, 4))
    ax[0].plot(p[0, :, 0])
    ax[0].plot(vad[0, :, 0])
    ax[0].axhline(0.5, color="k")
    for a in ax:
        a.set_ylim([0, 1.05])
    plt.pause(0.1)

    labels = vap_objective.extract_labels(vad)
    loss = vap_objective.loss_fn(logits, labels)
    print("-" * 45)
    print("vad: ", tuple(vad.shape))
    print("logits: ", tuple(logits.shape))
    print("labels: ", tuple(labels.shape))
    print("out['labels']: ", tuple(out["labels"].shape))
    print("out['probs']: ", tuple(out["probs"].shape))
    print("out['p']: ", tuple(out["p"].shape))
    print("out['p_bc']: ", tuple(out["p_bc"].shape))
    print("Loss: ", loss)
    print("-" * 45)
