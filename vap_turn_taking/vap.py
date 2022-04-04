import torch
import torch.nn as nn
from einops import rearrange
from typing import List

from vap_turn_taking.utils import vad_to_dialog_vad_states


def probs_ind_backchannel(probs):
    """

    Extract the probabilities associated with

    A:   |__|--|--|__|
    B:   |__|__|__|--|

    """
    bc_pred = []

    # Iterate over speakers
    for current_speaker, backchanneler in zip([1, 0], [0, 1]):
        # Between speaker diff
        # --------------------
        # Is the last bin of the "backchanneler" less probable than last bin of current speaker?
        last_a_lt_b = probs[..., backchanneler, -1] < probs[..., current_speaker, -1]

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


def bin_times_to_frames(bin_times, frame_hz):
    bt = torch.tensor(bin_times)
    return (bt * frame_hz).long().tolist()


class WindowHelper:
    @staticmethod
    def all_permutations_mono(n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    @staticmethod
    def end_of_segment_mono(n, max=3):
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

    @staticmethod
    def on_activity_change_mono(n=4, min_active=2):
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
            perms = WindowHelper.all_permutations_mono(permutable)
            base = base.repeat(perms.shape[0], 1)
            base[:, :permutable] = perms
        return base

    @staticmethod
    def combine_speakers(x1, x2, mirror=False):
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


class VAPLabel(nn.Module):
    def __init__(
        self,
        bin_times: List = [0.2, 0.4, 0.6, 0.8],
        frame_hz: int = 100,
        threshold_ratio: float = 0.5,
    ):
        super().__init__()
        self.bin_times = bin_times
        self.frame_hz = frame_hz
        self.threshold_ratio = threshold_ratio

        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_frames)

    def __repr__(self) -> str:
        s = "VAPLabel(\n"
        s += f"  bin_times: {self.bin_times}\n"
        s += f"  bin_frames: {self.bin_frames}\n"
        s += f"  frame_hz: {self.frame_hz}\n"
        s += f"  thresh: {self.threshold_ratio}\n"
        s += ")\n"
        return s

    def projection(self, va):
        """
        Extract projection (bins)
        (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames

        Arguments:
            va:         torch.Tensor (B, N, C)

        Returns:
            vaps:       torch.Tensor (B, m, C, M)

        """
        # Shift to get next frame projections
        return va[..., 1:, :].unfold(dimension=-2, size=sum(self.bin_frames), step=1)

    def vap_bins(self, projection_window):
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """

        start = 0
        v_bins = []
        for b in self.bin_frames:
            end = start + b
            m = projection_window[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        return torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)

    def comparative(self, projection_window):
        """
        Sum together the activity for each speaker in the `projection_window` and get the activity
        ratio for each speaker (focused on speaker 0)
        p(speaker_1) = 1 - p(speaker_0)
        vad:        torch.tensor, (B, N, 2)
        comp:       torch.tensor, (B, N)
        """
        comp = projection_window.sum(dim=-1)  # sum all activity for speakers
        tot = comp.sum(dim=-1) + 1e-9  # get total activity
        # focus on speaker 0 and get ratio: p(speaker_1)= 1 - p(speaker_0)
        comp = comp[..., 0] / tot
        return comp

    def __call__(self, va: torch.Tensor, type: str = "binary") -> torch.Tensor:
        projection_windows = self.projection(va)

        if type == "comparative":
            return self.comparative(projection_windows)

        return self.vap_bins(projection_windows)


class ActivityEmb(nn.Module):
    def __init__(self, bin_times=[0.20, 0.40, 0.60, 0.80], frame_hz=100):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_frames = bin_times_to_frames(bin_times, frame_hz)
        self.n_bins = len(self.bin_frames)
        self.total_bins = self.n_bins * 2
        self.n_classes = 2 ** self.total_bins

        # Discrete indices
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

        def single_idx_to_onehot(idx, d=8):
            assert idx < 2 ** d, "must be possible with {d} binary digits"
            z = torch.zeros(d)
            b = bin(idx).replace("0b", "")
            for i, v in enumerate(b[::-1]):
                z[i] = float(v)
            return z

        def create_code_vectors(n_bins):
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

    def init_subset_weighted_by_bin_size(self):
        """
        Calculates the comparative probability between the activity in each window for each speaker.

        a = sum(scale*activity_speaker_a)
        b = sum(scale*activity_speaker_b)
        p_a = a / (a+b)
        p_b = 1 - p_a
        """

        def oh_to_prob(oh):
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
        oh = scale_bins * self.idx_to_onehot(idx)
        subset_silence = oh_to_prob(oh)
        subset_active = oh_to_prob(oh[..., 2:])
        return subset_silence, subset_active

    def sort_idx(self, x):
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

    def init_subset_silence(self):
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = WindowHelper.combine_speakers(active, non_active, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self.sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_subset_active(self):
        """On activity"""
        # Shift subset
        eos = WindowHelper.end_of_segment_mono(self.n_bins, max=2)
        nav = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        shift_oh = WindowHelper.combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self.sort_idx(shift)

        # Don't shift subset
        eos = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        zero = torch.zeros((1, self.n_bins))
        hold_oh = WindowHelper.combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self.sort_idx(hold)
        return shift, hold

    def init_subset_backchannel(self, n=4):
        if n != 4:
            raise NotImplementedError("Not implemented for bin-size != 4")

        # at least 1 bin active over 3 bins
        bc_speaker = WindowHelper.all_permutations_mono(n=3, start=1)
        bc_speaker = torch.cat(
            (bc_speaker, torch.zeros((bc_speaker.shape[0], 1))), dim=-1
        )

        # all permutations of 3 bins
        current = WindowHelper.all_permutations_mono(n=3, start=0)
        current = torch.cat((current, torch.ones((current.shape[0], 1))), dim=-1)

        bc_both = WindowHelper.combine_speakers(bc_speaker, current, mirror=True)
        return self.onehot_to_idx(bc_both)

    def onehot_to_idx(self, x) -> torch.Tensor:
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

    def idx_to_onehot(self, idx):
        v = self.codebook(idx)
        return rearrange(v, "... (c b) -> ... c b", c=2)

    def forward(self, projection_window):
        return self.onehot_to_idx(projection_window)


class Probabilites:
    def _normalize_ind_probs(self, probs):
        probs = probs.sum(dim=-1)  # sum all bins for each speaker
        return probs / probs.sum(dim=-1, keepdim=True)  # norm

    def _marginal_probs(self, probs, pos_idx, neg_idx):
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def _silence_probs(self, p_a, p_b, sil_probs, silence):
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]
        return p_a, p_b

    def _single_speaker_probs(self, p0, p1, act_probs, current, other_speaker):
        w = torch.where(current)
        p0[w] = 1 - act_probs[w][..., other_speaker]  # P_a = 1-P_b
        p1[w] = act_probs[w][..., other_speaker]  # P_b
        return p0, p1

    def _overlap_probs(self, p_a, p_b, act_probs, both):
        """
        P_a_prior=A is next (active)
        P_b_prior=B is next (active)
        We the compare/renormalize given the two values of A/B is the next speaker
        sum = P_a_prior+P_b_prior
        P_a = P_a_prior / sum
        P_b = P_b_prior / sum
        """
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]

        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        return p_a, p_b


class VAP(nn.Module, Probabilites):
    TYPES = ["discrete", "independent", "comparative"]

    def __init__(
        self,
        type="discrete",
        bin_times=[0.20, 0.40, 0.60, 0.80],
        frame_hz=100,
        pre_frames=2,
        threshold_ratio=0.5,
    ):
        super().__init__()
        assert type in VAP.TYPES, "{type} is not a valid type! {VAP.TYPES}"

        self.type = type
        self.frame_hz = frame_hz
        self.bin_times = bin_times
        self.emb = ActivityEmb(bin_times, frame_hz)
        self.vap_label = VAPLabel(bin_times, frame_hz, threshold_ratio)
        self.horizon = torch.tensor(self.bin_times).sum(0).item()
        self.pre_frames = pre_frames

    @property
    def vap_bins(self):
        n = torch.arange(self.emb.n_classes, device=self.emb.codebook.weight.device)
        return self.emb.idx_to_onehot(n)

    def __repr__(self):
        s = super().__repr__().split("\n")
        s.insert(1, f"  type: {self.type}")
        s = "\n".join(s)
        return s

    def _probs_on_silence(self, probs):
        return self._marginal_probs(
            probs, self.emb.subset_silence, self.emb.subset_silence_hold
        )

    def _probs_on_active(self, probs):
        return self._marginal_probs(
            probs, self.emb.subset_active, self.emb.subset_active_hold
        )

    def _probs_ind_on_silence(self, probs):
        return self._normalize_ind_probs(probs)

    def _probs_ind_on_active(self, probs):
        return self._normalize_ind_probs(probs[..., :, self.pre_frames :])

    def _probs_weighted_on_silence(self, probs):
        sil_probs = probs.unsqueeze(
            -1
        ) * self.emb.subset_bin_weighted_silence.unsqueeze(0).to(probs.device)
        return sil_probs.sum(dim=-2)  # sum over classes

    def _probs_weighted_on_active(self, probs):
        # comparative active
        act_probs = probs.unsqueeze(-1) * self.emb.subset_bin_weighted_active.unsqueeze(
            0
        ).to(probs.device)
        return act_probs.sum(dim=-2)  # sum over classes

    def probs_backchannel(self, probs):
        ap = probs[..., self.emb.bc_prediction[0]].sum(-1)
        bp = probs[..., self.emb.bc_prediction[1]].sum(-1)
        return torch.stack((ap, bp), dim=-1)

    def probs_next_speaker(self, probs, va, type):
        """
        Extracts the probabilities for who the next speaker is dependent on what the current moment is.

        This means that on mutual silences we use the 'silence'-subset,
        where a single speaker is active we use the 'active'-subset and where on overlaps
        """
        if type == "independent":
            sil_probs = self._probs_ind_on_silence(probs)
            act_probs = self._probs_ind_on_active(probs)
        elif type == "weighted":
            sil_probs = self._probs_weighted_on_silence(probs)
            act_probs = self._probs_weighted_on_active(probs)
        else:  # discrete
            sil_probs = self._probs_on_silence(probs)
            act_probs = self._probs_on_active(probs)

        # Start wit all zeros
        # p_a: probability of A being next speaker (channel: 0)
        # p_b: probability of B being next speaker (channel: 1)
        p_a = torch.zeros_like(va[..., 0])
        p_b = torch.zeros_like(va[..., 0])

        # dialog states
        ds = vad_to_dialog_vad_states(va)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        p_a, p_b = self._silence_probs(p_a, p_b, sil_probs, silence)

        # A current speaker
        # Given only A is speaking we use the 'active' probability of B being the next speaker
        p_a, p_b = self._single_speaker_probs(
            p_a, p_b, act_probs, a_current, other_speaker=1
        )

        # B current speaker
        # Given only B is speaking we use the 'active' probability of A being the next speaker
        p_b, p_a = self._single_speaker_probs(
            p_b, p_a, act_probs, b_current, other_speaker=0
        )

        # Both
        p_a, p_b = self._overlap_probs(p_a, p_b, act_probs, both)

        p_probs = torch.stack((p_a, p_b), dim=-1)
        return p_probs

    def extract_label(self, va: torch.Tensor) -> torch.Tensor:
        if self.type == "comparative":
            return self.vap_label(va, type="comparative")

        vap_bins = self.vap_label(va, type="binary")

        if self.type == "independent":
            return vap_bins

        return self.emb(vap_bins)  # discrete

    def forward(self, logits, va):
        """
        Probabilites for events dependent on VAP-embedding and VA.
        """

        probs = logits.softmax(dim=-1)
        p = self.probs_next_speaker(probs=probs, va=va, type=self.type)

        # Next speaker probs
        if self.type == "discrete":
            p_bc = self.probs_backchannel(probs)
        elif self.type == "independent":
            # Backchannel probs (dependent on embedding and VA)
            p_bc = probs_ind_backchannel(probs)
        else:
            p_bc = None

        return {"p": p, "bc_prediction": p_bc}


if __name__ == "__main__":
    from vap_turn_taking.config.example_data import example, event_conf

    vapper = VAP(type="comparative")
    va = example["va"]
    y = vapper.extract_label(va)
    print("va: ", tuple(va.shape))
    print("y: ", tuple(y.shape))
    vapper
