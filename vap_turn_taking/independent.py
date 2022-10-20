import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

import vap_turn_taking.functional as VF
from vap_turn_taking.probabilities import Probs
from vap_turn_taking.projection_window import ProjectionWindow


class ComparativeVAP(nn.Module):
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.frame_hz = frame_hz
        self.projection_window_extractor = ProjectionWindow(
            bin_times, frame_hz, threshold_ratio
        )

    def probs_next_speaker(self, probs: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
        """
        Comparative does not have any way to change behavior given circumstances
        so `sil_probs == act_probs`
        """
        if probs.ndim == 2:
            probs = probs.unsqueeze(-1)
        p_b = 1 - probs
        sil_probs = act_probs = torch.cat((probs, p_b), dim=-1)
        print("probs: ", tuple(probs.shape))
        print("sil_probs: ", tuple(sil_probs.shape))
        print("act_probs: ", tuple(act_probs.shape))
        print("p_b: ", tuple(p_b.shape))
        p_next_speaker = Probs.next_speaker_probs(va, sil_probs, act_probs)
        return p_next_speaker

    def probs_backchannel(self):
        raise NotImplementedError(
            "Comparative objective does not have any backchannel probs"
        )

    def extract_labels(self, va: torch.Tensor) -> torch.Tensor:
        """
        Sum together the activity for each speaker in the `projection_window` and get the activity
        ratio for each speaker (focused on speaker 0)
        p(speaker_1) = 1 - p(speaker_0)
        vad:        torch.tensor, (B, N, 2)
        comp:       torch.tensor, (B, N)
        """
        proj_win = self.projection_window_extractor.projection(va)
        comp = proj_win.sum(dim=-1)  # sum all activity for speakers
        tot = comp.sum(dim=-1) + 1e-9  # get total activity
        # focus on speaker 0 and get ratio: p(speaker_1)= 1 - p(speaker_0)
        labels = comp[..., 0] / tot
        return labels

    def loss_fn(
        self, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:

        assert (
            labels.ndim == 2
        ), f"Exptected labels of shape (B, N_FRAMES) but got {labels.shape}"

        n_frames_logits = logits.shape[1]
        n_frames_labels = labels.shape[1]
        if n_frames_logits > n_frames_labels:
            logits = logits[:, :n_frames_labels]
        return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)

    def forward(
        self, logits: torch.Tensor, va: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, 2, N_BINS)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        # Labels (..., 2, self.n_bins)
        labels = self.extract_labels(va)

        # LABELS
        # Extract projection windows -> label indicies (B, N_VALID_FRAMES)
        n_valid_frames = labels.shape[-1]

        # Probs
        probs = logits[..., :n_valid_frames, :].sigmoid()
        p = self.probs_next_speaker(probs=probs, va=va[..., :n_valid_frames, :])
        return {"probs": probs, "p": p, "labels": labels}


class IndependentVAP(nn.Module):
    def __init__(
        self,
        bin_times: List[float] = [0.2, 0.4, 0.6, 0.8],
        pre_frames: int = 2,
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_frames = VF.bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.n_classes = 2 ** self.total_bins
        self.pre_frames = pre_frames

        # make projection windows
        self.projection_window_extractor = ProjectionWindow(
            bin_times, frame_hz, threshold_ratio
        )

    def normalize_probs(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.sum(dim=-1)  # sum all bins for each speaker
        return probs / probs.sum(dim=-1, keepdim=True)  # norm

    def probs_on_silence(self, probs: torch.Tensor) -> torch.Tensor:
        return self.normalize_probs(probs)

    def probs_on_active(self, probs: torch.Tensor, pre_frames: int) -> torch.Tensor:
        return self.normalize_probs(probs[..., :, pre_frames:])

    def probs_backchannel(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Extract the probabilities associated with the shap shown below

        E.g. a backhannel prediction for speaker A looks like:

        A:   |__|--|--|__|
        B:   |__|__|__|--|

        """
        bc_pred = []

        # Iterate over speakers
        for current_speaker, backchanneler in zip([1, 0], [0, 1]):
            # Between speaker diff
            # --------------------
            # Is the last bin of the "backchanneler" less probable than last bin of current speaker?
            last_a_lt_b = (
                probs[..., backchanneler, -1] < probs[..., current_speaker, -1]
            )

            # Within speaker diff
            # --------------------
            # Is the start/middle bins of the "backchanneler" greater than the last bin?
            # I.e. does it predict an ending response?
            mid_a_max, _ = probs[..., backchanneler, :-1].max(
                dim=-1
            )  # get prob (used with threshold)
            mid_a_gt_last = (
                mid_a_max > probs[..., backchanneler, -1]
            )  # find where the condition is true

            # Combine the between/within conditions
            non_zero_probs = torch.logical_and(last_a_lt_b, mid_a_gt_last)

            # Create probability tensor
            # P=0 where conditions are not met
            # P=max activation where condition is True
            tmp_pred_probs = torch.zeros_like(mid_a_max)
            tmp_pred_probs[non_zero_probs] = mid_a_max[non_zero_probs]
            bc_pred.append(tmp_pred_probs)

        bc_pred = torch.stack(bc_pred, dim=-1)
        return bc_pred

    def probs_next_speaker(self, probs: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
        """
        Extracts the probabilities for who the next speaker is dependent on what the current moment is.

        This means that on mutual silences we use the 'silence'-subset,
        where a single speaker is active we use the 'active'-subset and where on overlaps

        During active speech we omit the probabilities associated with the firts `pre_frames`
        bins for each speaker
        """
        sil_probs = self.probs_on_silence(probs)
        act_probs = self.probs_on_active(probs, self.pre_frames)

        p_next_speaker = Probs.next_speaker_probs(va, sil_probs, act_probs)
        return p_next_speaker

    def extract_labels(self, va: torch.Tensor) -> torch.Tensor:
        return self.projection_window_extractor(va)

    def loss_fn(
        self, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:

        assert (
            logits.ndim == 4
        ), f"Exptected logits of shape (B, N_FRAMES, 2, N_BINS) but got {logits.shape}"
        assert logits.shape[-2:] == (
            2,
            self.n_bins,
        ), f"Exptected logits of shape (B, N_FRAMES, 2, {self.n_bins}) but got {logits.shape}"

        assert (
            labels.ndim == 4
        ), f"Exptected labels of shape (B, N_FRAMES, 2, N_BINS) but got {labels.shape}"
        assert labels.shape[-2:] == (
            2,
            self.n_bins,
        ), f"Exptected labels of shape (B, N_FRAMES, 2, {self.n_bins}) but got {labels.shape}"

        n_frames_logits = logits.shape[1]
        n_frames_labels = labels.shape[1]

        if n_frames_logits > n_frames_labels:
            logits = logits[:, :n_frames_labels]

        return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)

    def forward(
        self, logits: torch.Tensor, va: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, 2, N_BINS)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        # Labels (..., 2, self.n_bins)
        labels = self.extract_labels(va)

        assert (
            logits.shape[-1] == self.n_bins
        ), f"Logits have wrong shape. {logits.shape} != (..., 2, {self.n_bins}) that is (B, N_FRAMES, 2, N_BINS)"

        # LABELS
        # Extract projection windows -> label indicies (B, N_VALID_FRAMES)
        n_valid_frames = labels.shape[-1]

        # Probs  BCE -> sigmoid instead of softmax
        probs = logits[..., :n_valid_frames, :].sigmoid()
        p = self.probs_next_speaker(probs=probs, va=va[..., :n_valid_frames, :])
        p_bc = self.probs_backchannel(probs)
        return {"probs": probs, "p": p, "p_bc": p_bc, "labels": labels}


if __name__ == "__main__":

    data = torch.load("example/vap_data.pt")

    vap_objective = IndependentVAP()
    vad = data["shift"]["vad"]
    logits = torch.randn((1, 500, 2, vap_objective.n_bins))
    out = vap_objective(logits, vad)
    labels = vap_objective.extract_labels(vad)
    loss = vap_objective.loss_fn(logits, labels)
    print("vad: ", tuple(vad.shape))
    print("logits: ", tuple(logits.shape))
    print("labels: ", tuple(labels.shape))
    print("out['labels']: ", tuple(out["labels"].shape))
    print("out['probs']: ", tuple(out["probs"].shape))
    print("out['p']: ", tuple(out["p"].shape))
    print("out['p_bc']: ", tuple(out["p_bc"].shape))
    print("Loss: ", loss)

    vap_objective = ComparativeVAP()
    vad = data["shift"]["vad"]
    logits = torch.randn((1, 500))
    out = vap_objective(logits, vad)
    labels = vap_objective.extract_labels(vad)
    loss = vap_objective.loss_fn(logits, labels)
    print("vad: ", tuple(vad.shape))
    print("logits: ", tuple(logits.shape))
    print("labels: ", tuple(labels.shape))
    print("out['labels']: ", tuple(out["labels"].shape))
    print("out['probs']: ", tuple(out["probs"].shape))
    print("out['p']: ", tuple(out["p"].shape))
