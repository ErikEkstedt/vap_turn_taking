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


if __name__ == "__main__":

    import torch

    data = torch.load("example/vap_data.pt")

    vap_objective = VAP()
    print()
    print(vap_objective)
    vad = data["shift"]["vad"]
    logits = torch.rand((1, 600, 256)) / 256
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
