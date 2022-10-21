import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Dict, List, Tuple

from vap_turn_taking.probabilities import Probs
from vap_turn_taking.projection_window import ProjectionWindow
import vap_turn_taking.functional as VF


def all_permutations_mono(n: int, start: int = 0) -> torch.Tensor:
    vectors = []
    for i in range(start, 2 ** n):
        i = bin(i).replace("0b", "").zfill(n)
        tmp = torch.zeros(n)
        for j, val in enumerate(i):
            tmp[j] = float(val)
        vectors.append(tmp)
    return torch.stack(vectors)


def end_of_segment_mono(n: int, max: int = 3) -> torch.Tensor:
    """
    # 0, 0, 0, 0
    # 1, 0, 0, 0
    # 1, 1, 0, 0
    # 1, 1, 1, 0
    """
    v = torch.zeros((max + 1, n))
    for i in range(max):
        v[i + 1, : i + 1] = 1
    return v


def on_activity_change_mono(n: int = 4, min_active: int = 2) -> torch.Tensor:
    """

    Used where a single speaker is active. This vector (single speaker) represents
    the classes we use to infer that the current speaker will end their activity
    and the other take over.

    the `min_active` variable corresponds to the minimum amount of frames that must
    be active AT THE END of the projection window (for the next active speaker).
    This used to not include classes where the activity may correspond to a short backchannel.
    e.g. if only the last bin is active it may be part of just a short backchannel, if we require 2 bins to
    be active we know that the model predicts that the continuation will be at least 2 bins long and thus
    removes the ambiguouty (to some extent) about the prediction.
    """

    base = torch.zeros(n)
    # force activity at the end
    if min_active > 0:
        base[-min_active:] = 1

    # get all permutations for the remaining bins
    permutable = n - min_active
    if permutable > 0:
        perms = all_permutations_mono(permutable)
        base = base.repeat(perms.shape[0], 1)
        base[:, :permutable] = perms
    return base


def combine_speakers(
    x1: torch.Tensor, x2: torch.Tensor, mirror: bool = False
) -> torch.Tensor:
    if x1.ndim == 1:
        x1 = x1.unsqueeze(0)
    if x2.ndim == 1:
        x2 = x2.unsqueeze(0)
    vad = []
    for a in x1:
        for b in x2:
            vad.append(torch.stack((a, b), dim=0))

    vad = torch.stack(vad)
    if mirror:
        vad = torch.stack((vad, torch.stack((vad[:, 1], vad[:, 0]), dim=1)))
    return vad


def sort_idx(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x, _ = x.sort()
    elif x.ndim == 2:
        if x.shape[0] == 2:
            a, _ = x[0].sort()
            b, _ = x[1].sort()
            x = torch.stack((a, b))
        else:
            x, _ = x[0].sort()
            x = x.unsqueeze(0)
    return x


class DiscreteVAP(nn.Module):
    def __init__(
        self,
        bin_times: List[float] = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 50,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_frames: List[int] = VF.bin_times_to_frames(bin_times, frame_hz)
        self.n_bins: int = len(self.bin_frames)
        self.total_bins: int = self.n_bins * 2
        self.n_classes: int = 2 ** self.total_bins

        # make projection windows
        self.projection_window_extractor = ProjectionWindow(
            bin_times, frame_hz, threshold_ratio
        )

        # Create all relevant subset for the DISCRETE representation
        self.codebook = self.init_codebook()

        # weighted by bin size (subset for active/silent is modified dependent on bin_frames)
        wsil, wact = self.init_subset_weighted_by_bin_size()
        self.subset_bin_weighted_silence = wsil
        self.subset_bin_weighted_active = wact

        self.subset_silence, self.subset_silence_hold = self.init_subset_silence()
        self.subset_active, self.subset_active_hold = self.init_subset_active()
        self.bc_prediction = self.init_subset_backchannel()
        self.requires_grad_(False)

    def init_codebook(self) -> nn.Module:
        """
        Initializes the codebook for the vad-projection horizon labels.

        Map all vectors of binary digits of length `n_bins` to their corresponding decimal value.

        This allows a VAD future of shape (*, 4, 2) to be flatten to (*, 8) and mapped to a number
        corresponding to the class index.
        """

        def single_idx_to_onehot(idx: int, d: int = 8) -> torch.Tensor:
            assert idx < 2 ** d, "must be possible with {d} binary digits"
            z = torch.zeros(d)
            b = bin(idx).replace("0b", "")
            for i, v in enumerate(b[::-1]):
                z[i] = float(v)
            return z

        def create_code_vectors(n_bins: int) -> torch.Tensor:
            """
            Create a matrix of all one-hot encodings representing a binary sequence of `self.total_bins` places
            Useful for usage in `nn.Embedding` like module.
            """
            n_codes = 2 ** n_bins
            embs = torch.zeros((n_codes, n_bins))
            for i in range(2 ** n_bins):
                embs[i] = single_idx_to_onehot(i, d=n_bins)
            return embs

        codebook = nn.Embedding(
            num_embeddings=self.n_classes, embedding_dim=self.total_bins
        )
        codebook.weight.data = create_code_vectors(self.total_bins)
        codebook.weight.requires_grad_(False)
        return codebook

    def init_subset_weighted_by_bin_size(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the comparative probability between the activity in each window for each speaker.

        a = sum(scale*activity_speaker_a)
        b = sum(scale*activity_speaker_b)
        p_a = a / (a+b)
        p_b = 1 - p_a
        """

        def oh_to_prob(oh: torch.Tensor) -> torch.Tensor:
            tot = oh.sum(dim=-1).sum(dim=-1)
            a_comp = oh[:, 0].sum(-1) / (tot + 1e-9)
            # No activity counts as equal
            a_comp[a_comp == 0] = 0.5
            b_comp = 1 - a_comp
            return torch.stack((a_comp, b_comp), dim=-1)

        # get all onehot-states
        idx = torch.arange(self.n_classes)

        # normalize bin size weights -> adds to one
        scale_bins = torch.tensor(self.bin_frames, dtype=torch.float)
        scale_bins /= scale_bins.sum()

        # scale the bins of the onehot-states
        oh = scale_bins * self.idx_to_proj_win(idx)
        subset_silence = oh_to_prob(oh)
        subset_active = oh_to_prob(oh[..., 2:])
        return subset_silence, subset_active

    def init_subset_silence(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = on_activity_change_mono(self.n_bins, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = combine_speakers(active, non_active, mirror=True)
        shift = self.proj_win_to_idx(shift_oh)
        shift = sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_subset_active(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """On activity"""
        # Shift subset
        eos = end_of_segment_mono(self.n_bins, max=2)
        nav = on_activity_change_mono(self.n_bins, min_active=2)
        shift_oh = combine_speakers(nav, eos, mirror=True)
        shift = self.proj_win_to_idx(shift_oh)
        shift = sort_idx(shift)

        # Don't shift subset
        eos = on_activity_change_mono(self.n_bins, min_active=2)
        zero = torch.zeros((1, self.n_bins))
        hold_oh = combine_speakers(zero, eos, mirror=True)
        hold = self.proj_win_to_idx(hold_oh)
        hold = sort_idx(hold)
        return shift, hold

    def init_subset_backchannel(self, n: int = 4) -> torch.Tensor:
        if n != 4:
            raise NotImplementedError("Not implemented for bin-size != 4")

        # at least 1 bin active over 3 bins
        bc_speaker = all_permutations_mono(n=3, start=1)
        bc_speaker = torch.cat(
            (bc_speaker, torch.zeros((bc_speaker.shape[0], 1))), dim=-1
        )

        # all permutations of 3 bins
        current = all_permutations_mono(n=3, start=0)
        current = torch.cat((current, torch.ones((current.shape[0], 1))), dim=-1)

        bc_both = combine_speakers(bc_speaker, current, mirror=True)
        return self.proj_win_to_idx(bc_both)

    def probs_on_silence(self, probs: torch.Tensor) -> torch.Tensor:
        return Probs.marginal_probs(
            probs, self.subset_silence, self.subset_silence_hold
        )

    def probs_on_active(self, probs: torch.Tensor) -> torch.Tensor:
        return Probs.marginal_probs(probs, self.subset_active, self.subset_active_hold)

    def probs_backchannel(self, probs: torch.Tensor) -> torch.Tensor:
        ap = probs[..., self.bc_prediction[0]].sum(-1)
        bp = probs[..., self.bc_prediction[1]].sum(-1)
        return torch.stack((ap, bp), dim=-1)

    def probs_next_speaker(self, probs: torch.Tensor, va: torch.Tensor) -> torch.Tensor:
        """
        Extracts the probabilities for who the next speaker is dependent on what the current moment is.

        This means that on mutual silences we use the 'silence'-subset,
        where a single speaker is active we use the 'active'-subset and where on overlaps
        """
        sil_probs = self.probs_on_silence(probs)
        act_probs = self.probs_on_active(probs)
        p_next_speaker = Probs.next_speaker_probs(va, sil_probs, act_probs)
        return p_next_speaker

    def proj_win_to_idx(self, x: torch.Tensor) -> torch.Tensor:
        """
        The inverse of the 'forward' function.

        Arguments:
            x:          torch.Tensor (*, 2, 4)

        Inspiration for distance calculation:
            https://github.com/lucidrains/vector-quantize-pytorch/blob/master/vector_quantize_pytorch/vector_quantize_pytorch.py
        """
        assert x.shape[-2:] == (2, self.n_bins)

        # compare with codebook and get closest idx
        shape = x.shape
        flatten = rearrange(x, "... c bpp -> (...) (c bpp)", c=2, bpp=self.n_bins)
        embed = self.codebook.weight.t()

        dist = -(
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        embed_ind = embed_ind.view(*shape[:-2])
        return embed_ind

    def idx_to_proj_win(self, idx: torch.Tensor) -> torch.Tensor:
        v = self.codebook(idx)
        return rearrange(v, "... (c b) -> ... c b", c=2)

    def extract_labels(self, va: torch.Tensor) -> torch.Tensor:
        projection_windows = self.projection_window_extractor(va)
        return self.proj_win_to_idx(projection_windows)

    def loss_fn(
        self, logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        assert (
            logits.ndim == 3
        ), f"Exptected logits of shape (B, N_FRAMES, N_CLASSES) but got {logits.shape}"
        assert (
            labels.ndim == 2
        ), f"Exptected labels of shape (B, N_FRAMES) but got {labels.shape}"

        n_frames_logits = logits.shape[1]
        n_frames_labels = labels.shape[1]

        if n_frames_logits > n_frames_labels:
            logits = logits[:, :n_frames_labels]

        # CrossEntropyLoss over discrete labels
        loss = F.cross_entropy(
            rearrange(logits, "b n d -> (b n) d"),
            rearrange(labels, "b n -> (b n)"),
            reduction=reduction,
        )

        # Shape back to original shape if reduction != 'none'
        if reduction == "none":
            n = logits.shape[1]
            loss = rearrange(loss, "(b n) -> b n", n=n)
        return loss

    def forward(
        self, logits: torch.Tensor, va: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extracts labels from the voice-activity, va.
        The labels are based on projections of the future and so the valid
        frames with corresponding labels are strictly less then the original number of frams.

        Arguments:
        -----------
        logits:     torch.Tensor (B, N_FRAMES, N_CLASSES)
        va:         torch.Tensor (B, N_FRAMES, 2)

        Return:
        -----------
            Dict[probs, p, p_bc, labels]  which are all torch.Tensors
        """

        assert (
            logits.shape[-1] == self.n_classes
        ), f"Logits have wrong shape. {logits.shape} != (..., {self.n_classes}) that is (B, N_FRAMES, N_CLASSES)"

        # LABELS
        # Extract projection windows -> label indicies (B, N_VALID_FRAMES)
        labels = self.extract_labels(va)
        n_valid_frames = labels.shape[-1]

        # Probs
        probs = logits[..., :n_valid_frames, :].softmax(dim=-1)
        p = self.probs_next_speaker(probs=probs, va=va[..., :n_valid_frames, :])
        p_bc = self.probs_backchannel(probs)
        return {"probs": probs, "p": p, "p_bc": p_bc, "labels": labels}


if __name__ == "__main__":

    data = torch.load("example/vap_data.pt")

    vap_objective = DiscreteVAP()
    vad = data["shift"]["vad"]
    logits = torch.rand((1, 600, 256)) / 256
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
