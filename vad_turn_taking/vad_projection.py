import torch
import torch.nn as nn
from einops import rearrange

from vad_turn_taking.utils import time_to_frames
from vad_turn_taking.vad import VAD


def add_start_end(x, val=[0], start=True):
    n = x.shape[0]
    out = []
    for v in val:
        pad = torch.ones(n) * v
        if start:
            o = torch.cat((pad.unsqueeze(1), x), dim=-1)
        else:
            o = torch.cat((x, pad.unsqueeze(1)), dim=-1)
        out.append(o)
    return torch.cat(out)


class VadLabel:
    def __init__(self, bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=100, threshold_ratio=0.5):
        self.bin_times = bin_times
        self.vad_hz = vad_hz
        self.threshold_ratio = threshold_ratio

        self.bin_sizes = time_to_frames(bin_times, vad_hz)
        self.n_bins = len(self.bin_sizes)
        self.total_bins = self.n_bins * 2
        self.horizon = sum(self.bin_sizes)

    def horizon_to_onehot(self, vad_projections):
        """
        Iterate over the bin boundaries and sum the activity
        for each channel/speaker.
        divide by the number of frames to get activity ratio.
        If ratio is greater than or equal to the threshold_ratio
        the bin is considered active
        """
        start = 0
        v_bins = []
        for b in self.bin_sizes:
            end = start + b
            m = vad_projections[..., start:end].sum(dim=-1) / b
            m = (m >= self.threshold_ratio).float()
            v_bins.append(m)
            start = end
        v_bins = torch.stack(v_bins, dim=-1)  # (*, t, c, n_bins)
        # Treat the 2-channel activity as a single binary sequence
        v_bins = v_bins.flatten(-2)  # (*, t, c, n_bins) -> (*, t, (c n_bins))
        return rearrange(v_bins, "... (c d) -> ... c d", c=2)

    def vad_projection(self, vad) -> torch.Tensor:
        """
        Given a sequence of binary VAD information (two channels) we extract a prediction horizon
        (frame length = the sum of all bin_sizes).

        ! WARNING ! VAD is shifted one step to get the 'next frame horizon'

        ```python
        # vad: (B, N, 2)
        # DO THIS
        vad_projection_oh = VadProjection.vad_to_idx(vad)
        # vad_projection_oh: (B, N, 2, )
        ```

        Arguments:
            vad:        torch.Tensor, (b, n, c) or (n, c)
        """
        # (b, n, c) -> (b, N, c, M), M=horizon window size, N=valid frames
        # Shift to get next frame projections
        vv = vad[..., 1:, :]
        vad_projections = vv.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)

        # (b, N, c, M) -> (B, N, 2, len(self.bin_sizes))
        v_bins = self.horizon_to_onehot(vad_projections)
        return v_bins

    def comparative_activity(self, vad):
        """
        Sum together the activity for each speaker in the `projection_window` and get the activity
        ratio for each speaker (focused on speaker 0)
        p(speaker_1) = 1 - p(speaker_0)
        vad:        torch.tensor, (B, N, 2)
        comp:       torch.tensor, (B, N)
        """
        vv = vad[..., 1:, :]
        projection_windows = vv.unfold(dimension=-2, size=sum(self.bin_sizes), step=1)
        comp = projection_windows.sum(dim=-1)  # sum all activity for speakers
        tot = comp.sum(dim=-1) + 1e-9  # get total activity
        # focus on speaker 0 and get ratio: p(speaker_1)= 1 - p(speaker_0)
        comp = comp[..., 0] / tot
        return comp


class ProjectionCodebook(nn.Module):
    def __init__(self, bin_times=[0.20, 0.40, 0.60, 0.80], frame_hz=100):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_sizes = time_to_frames(bin_times, frame_hz)

        self.n_bins = len(bin_times)
        self.total_bins = self.n_bins * 2
        self.n_classes = 2 ** self.total_bins

        self.codebook = self.init_codebook()
        self.comparative_probabilities = self.init_comparative_classes()
        self.on_silent_shift, self.on_silent_hold = self.init_on_silent_shift()
        self.on_silent_next = self.on_silent_shift
        self.on_active_shift, self.on_active_hold = self.init_on_activity_shift()
        self.bc_active = self.init_bc_prediction(self.n_bins)
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

    def _all_permutations_mono(self, n, start=0):
        vectors = []
        for i in range(start, 2 ** n):
            i = bin(i).replace("0b", "").zfill(n)
            tmp = torch.zeros(n)
            for j, val in enumerate(i):
                tmp[j] = float(val)
            vectors.append(tmp)
        return torch.stack(vectors)

    def _end_of_segment_mono(self, n, max=3):
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

    def _on_activity_change_mono(self, n=4, min_active=2):
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
            perms = self._all_permutations_mono(permutable)
            base = base.repeat(perms.shape[0], 1)
            base[:, :permutable] = perms
        return base

    def _combine_speakers(self, x1, x2, mirror=False):
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

    def _sort_idx(self, x):
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

    def init_comparative_classes(self):
        """
        Calculates the comparative probability between the activity in each window for each speaker.

        a = sum(activity_speaker_a)
        b = sum(activity_speaker_b)
        p_a = a / (a+b)
        p_b = 1 - p_a
        """
        idx = torch.arange(self.n_classes)
        oh = self.idx_to_onehot(idx)
        # Extract the comparative probability for each class
        # first is all zeros and set to equal chance
        no_activity_prob = torch.tensor([0.5])
        tot = oh[:, 0].sum(-1) + oh[:, 1].sum(-1)
        a_comp = oh[1:, 0].sum(-1) / tot[1:]
        a_comp = torch.cat([no_activity_prob, a_comp])
        b_comp = 1 - a_comp
        return torch.stack((a_comp, b_comp), dim=-1)

    def init_on_silent_shift(self):
        """
        During mutual silences we wish to infer which speaker the model deems most likely.

        We focus on classes where only a single speaker is active in the projection window,
        renormalize the probabilities on this subset, and determine which speaker is the most
        likely next speaker.
        """

        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        # active = self._all_permutations_mono(n, start=1)  # at least 1 active
        # active channel: At least 1 bin is active -> all permutations (all except the no-activity)
        active = self._on_activity_change_mono(self.n_bins, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = self._combine_speakers(active, non_active, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_on_activity_shift(self):
        # Shift subset
        eos = self._end_of_segment_mono(self.n_bins, max=2)
        nav = self._on_activity_change_mono(self.n_bins, min_active=2)
        shift_oh = self._combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # Don't shift subset
        eos = self._on_activity_change_mono(self.n_bins, min_active=2)
        zero = torch.zeros((1, self.n_bins))
        hold_oh = self._combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self._sort_idx(hold)
        return shift, hold

    def init_bc_prediction(self, n=4):
        if n != 4:
            raise NotImplementedError("Not implemented for bin-size != 4")

        # after second
        bc = [0, 1, 0, 0]
        cur = self._all_permutations_mono(n=2, start=1)
        cur = add_start_end(cur, val=[0, 1])
        cur = add_start_end(cur, val=[0, 1])
        bc2 = self._combine_speakers(torch.tensor(bc), cur)
        # after third
        bc = [[0, 1, 1, 0], [0, 0, 1, 0]]
        cur = self._all_permutations_mono(n=3, start=0)
        cur = add_start_end(cur, val=[1], start=False)
        bc3 = self._combine_speakers(torch.tensor(bc), cur)
        bc_cur = torch.cat((bc2, bc3))
        bc_cur2 = bc_cur.flip(1)

        bc_both = torch.stack((bc_cur, bc_cur2))
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

    def get_marginal_probs(self, probs, pos_idx, neg_idx):
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def get_silence_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_silent_shift, self.on_silent_hold)

    def get_active_shift_probs(self, probs):
        return self.get_marginal_probs(probs, self.on_active_shift, self.on_active_hold)

    def get_next_speaker_probs(self, logits, vad):
        probs = logits.softmax(dim=-1)
        sil_probs = self.get_silence_shift_probs(probs)
        act_probs = self.get_active_shift_probs(probs)

        p_a = torch.zeros_like(sil_probs[..., 0])
        p_b = torch.zeros_like(sil_probs[..., 0])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]

        # A current speaker
        w = torch.where(a_current)
        p_b[w] = act_probs[w][..., 1]
        p_a[w] = 1 - act_probs[w][..., 1]

        # B current speaker
        w = torch.where(b_current)
        p_a[w] = act_probs[w][..., 0]
        p_b[w] = 1 - act_probs[w][..., 0]

        # Both
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]
        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum
        return torch.stack((p_a, p_b), dim=-1)

    def speaker_prob_to_shift(self, probs, vad):
        """
        next speaker probabilities (B, N, 2) -> turn-shift probabilities (B, n)
        """
        assert probs.ndim == 3, "Assumes probs.shape = (B, N, 2)"

        shift_probs = torch.zeros(probs.shape[:-1])

        # dialog states
        ds = VAD.vad_to_dialog_vad_states(vad)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        prev_speaker = VAD.get_last_speaker(vad)

        # A active -> B = 1 is next_speaker
        w = torch.where(a_current)
        shift_probs[w] = probs[w][..., 1]
        # B active -> A = 0 is next_speaker
        w = torch.where(b_current)
        shift_probs[w] = probs[w][..., 0]
        # silence and A was previous speaker -> B = 1 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 0))
        shift_probs[w] = probs[w][..., 1]
        # silence and B was previous speaker -> A = 0 is next_speaker
        w = torch.where(torch.logical_and(silence, prev_speaker == 1))
        shift_probs[w] = probs[w][..., 0]
        return shift_probs

    def forward(self, projection_window):
        # return self.idx_to_onehot(idx)
        return self.onehot_to_idx(projection_window)


def time_label_making():
    import time

    vad = torch.randint(0, 2, (128, 1000, 2))

    FRAME_HZ = 100

    VL = VadLabel(bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=FRAME_HZ)
    # Time label making
    t = time.time()
    for i in range(10):
        lab_oh = VL.vad_projection(vad)
    t = time.time() - t
    print("bin_times: ", len(VL.bin_times), " took: ", round(t, 4), "seconds")

    VL = VadLabel(bin_times=[0.05] * 60, vad_hz=FRAME_HZ)
    # Time label making
    t = time.time()
    for i in range(10):
        lab_oh = VL.vad_projection(vad)
    t = time.time() - t
    print("bin_times: ", len(VL.bin_times), " took: ", round(t, 4), "seconds")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import math
    from vad_turn_taking.plot_utils import (
        plot_events,
        plot_vad_oh,
        plot_projection_window,
    )
    from vad_turn_taking import DialogEvents

    from conv_ssl.evaluation.utils import load_model, load_dm

    dm = load_dm()
    diter = iter(dm.val_dataloader())

    codebook = ProjectionCodebook()
    print("bc: ", codebook.bc_active.shape)
    print("sil shift: ", codebook.on_silent_shift.shape)
    print("act shift: ", codebook.on_active_shift.shape)

    next_a = codebook.idx_to_onehot(codebook.on_silent_next[0])
    next_b = codebook.idx_to_onehot(codebook.on_silent_next[1])

    a = codebook._end_of_segment_mono(codebook.n_bins, max=2)

    # Always a Backchannel prediction
    bc = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 1, 0]]
    bc0 = codebook._combine_speakers(torch.tensor(bc), torch.tensor([1, 1, 1, 1]))
    active = [[1, 1, 1, 0], [1, 0, 1, 0]]
    bc1 = codebook._combine_speakers(torch.tensor([0, 1, 0, 0]), torch.tensor(active))
    active = [[0, 1, 1, 1], [0, 1, 0, 1]]
    bc2 = codebook._combine_speakers(torch.tensor([0, 0, 1, 0]), torch.tensor(active))
    bc = torch.cat((bc0, bc1, bc2))

    fig, ax = plt.subplots(bc.shape[0], 1, sharex=True)
    for i, bb in enumerate(bc):
        plot_projection_window(bb, ax=ax[i])  # , bin_frames=[20, 40, 60, 80])
    plt.pause(0.1)

    # Backchannel prediction
    # bc = [[1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0,0,1,0]]

    bc_cur = codebook.idx_to_onehot(codebook.bc_active[0])
    cols = 5
    rows = math.ceil(bc_cur.shape[0] / cols)
    fig, ax = plt.subplots(rows, cols, sharex=True)
    n = 0
    for col in range(cols):
        for row in range(rows):
            if n >= bc_cur.shape[0]:
                break
            bb = bc_cur[n]
            # plot_projection_window(bb, ax=ax[row, col]) #, bin_frames=[20, 40, 60, 80])
            plot_projection_window(bb, ax=ax[row, col], bin_frames=[20, 40, 60, 80])
            n += 1
    plt.pause(0.1)

    # Backchannel contingent on current speaker
    fig, ax = plt.subplots(bc.shape[0], 1, sharex=True)
    for i, bb in enumerate(bc_bcur):
        plot_projection_window(bb, ax=ax[i])  # , bin_frames=[20, 40, 60, 80])
    plt.pause(0.1)

    # non-active channel: zeros
    non_active = torch.zeros((1, active.shape[-1]))
    # combine
    shift_oh = self._combine_speakers(active, non_active, mirror=True)

    codebook._all_permutations_mono(n=4, start=2)

    fig, ax = plt.subplots(len(next_a), 1, sharex=True)
    for i, oh in enumerate(next_a):
        plot_projection_window(oh, ax=ax[i], bin_frames=[20, 40, 60, 80])
        # plot_projection_window(oh, ax=ax[i])
    figb, axb = plt.subplots(len(next_a), 1, sharex=True)
    for i, oh in enumerate(next_b):
        plot_projection_window(oh, ax=axb[i], bin_frames=[20, 40, 60, 80])
        # plot_projection_window(oh, ax=axb[i])
    plt.pause(0.1)

    codebook.on_silent_hold
    codebook.on_active_shift
    codebook.on_active_hold

    a_next = codebook.idx_to_onehot(codebook.on_silent_shift)

    # Shift/Hold params
    start_pad = 5
    min_context = 50
    active_frames = 50
    # Backchannel params
    bc_pre_silence_frames = 150  # 1.5 seconds
    bc_post_silence_frames = 300  # 3 seconds
    bc_max_active_frames = 200  # 2 seconds

    batch = next(diter)
    vad = batch["vad"]  # vad: (B, N, 2)
    print("vad: ", tuple(vad.shape))
    vad = batch["vad"]
    # valid
    hold, shift = DialogEvents.on_silence(
        vad,
        start_pad=start_pad,
        target_frames=50,
        horizon=100,
        min_context=min_context,
    )
    # Find active segment pre-events
    pre_hold, pre_shift = DialogEvents.get_active_pre_events(
        vad,
        hold,
        shift,
        start_pad=start_pad,
        active_frames=active_frames,
        min_context=min_context,
    )
    backchannels = DialogEvents.extract_bc_candidates(
        vad,
        pre_silence_frames=bc_pre_silence_frames,
        post_silence_frames=bc_post_silence_frames,
        max_active_frames=bc_max_active_frames,
    )
    print("hold: ", tuple(hold.shape))
    print("pre_hold: ", tuple(pre_hold.shape))
    print("shift: ", tuple(shift.shape))
    print("pre_shift: ", tuple(pre_shift.shape))
    print("backchannels: ", tuple(backchannels.shape))
    ev = torch.logical_or(backchannels[..., 0], backchannels[..., 1])

    for b in range(4):
        fig, ax = plot_events(
            vad[b], hold=pre_hold[b], shift=pre_shift[b], event=ev[b], event_alpha=0.2
        )
