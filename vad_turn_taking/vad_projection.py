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


class ProjectionCodebook(nn.Module):
    def __init__(self, bin_times=[0.20, 0.40, 0.60, 0.80], frame_hz=100):
        super().__init__()
        self.frame_hz = frame_hz
        self.bin_sizes = time_to_frames(bin_times, frame_hz)

        self.n_bins = len(bin_times)
        self.total_bins = self.n_bins * 2
        self.n_classes = 2 ** self.total_bins

        self.codebook = self.init_codebook()
        self.comp_silent, self.comp_active = self.init_comparative_classes()
        self.on_silent_shift, self.on_silent_hold = self.init_on_silent_shift()
        self.on_silent_next = self.on_silent_shift
        self.on_active_shift, self.on_active_hold = self.init_on_activity_shift()
        self.bc_prediction = self.init_bc_prediction()
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
        scale_bins = torch.tensor(self.bin_sizes, dtype=torch.float)
        scale_bins /= scale_bins.sum()

        # scale the bins of the onehot-states
        oh = scale_bins * self.idx_to_onehot(idx)
        comp_silent = oh_to_prob(oh)
        comp_active = oh_to_prob(oh[..., 2:])
        return comp_silent, comp_active

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
        active = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        # non-active channel: zeros
        non_active = torch.zeros((1, active.shape[-1]))
        # combine
        shift_oh = WindowHelper.combine_speakers(active, non_active, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # symmetric, this is strictly unneccessary but done for convenience and to be similar
        # to 'get_on_activity_shift' where we actually have asymmetric classes for hold/shift
        hold = shift.flip(0)
        return shift, hold

    def init_on_activity_shift(self):
        """On activity"""
        # Shift subset
        eos = WindowHelper.end_of_segment_mono(self.n_bins, max=2)
        nav = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        shift_oh = WindowHelper.combine_speakers(nav, eos, mirror=True)
        shift = self.onehot_to_idx(shift_oh)
        shift = self._sort_idx(shift)

        # Don't shift subset
        eos = WindowHelper.on_activity_change_mono(self.n_bins, min_active=2)
        zero = torch.zeros((1, self.n_bins))
        hold_oh = WindowHelper.combine_speakers(zero, eos, mirror=True)
        hold = self.onehot_to_idx(hold_oh)
        hold = self._sort_idx(hold)
        return shift, hold

    def init_bc_prediction(self, n=4):
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

    def get_marginal_probs(self, probs, pos_idx, neg_idx):
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    def next_speaker_probs(self, probs, silence=False):
        if silence:
            return self.get_marginal_probs(probs, self.on_silent_shift, self.on_silent_hold)
        else:
            return self.get_marginal_probs(probs, self.on_active_shift, self.on_active_hold)

    def get_next_speaker_probs(self, logits=None, vad=None, probs=None, pw=False):
        """
        Extracts the probabilities for who the next speaker is dependent on what the current moment is.

        This means that on mutual silences we use the 'silence'-subset,
        where a single speaker is active we use the 'active'-subset and where on overlaps
        """

        if probs is None:
            probs = logits.softmax(dim=-1)

        sil_probs = self.next_speaker_probs(probs, silence=True)
        act_probs = self.next_speaker_probs(probs, silence=False)
        # sil_probs = self.get_silence_shift_probs(probs)
        # act_probs = self.get_active_shift_probs(probs)

        # Start wit all zeros
        # p_a: probability of A being next speaker (channel: 0)
        # p_b: probability of B being next speaker (channel: 1)
        p_a = torch.zeros_like(sil_probs[..., 0])
        p_b = torch.zeros_like(sil_probs[..., 0])

        pw_a = None
        pw_b = None

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

        if pw:
            pw_a = torch.zeros_like(sil_probs[..., 0])
            pw_b = torch.zeros_like(sil_probs[..., 0])

            # comparative silent
            comp_sil = probs.unsqueeze(-1) * self.comp_silent.unsqueeze(0).to(
                probs.device
            )
            comp_sil = comp_sil.sum(dim=-2)  # sum over classes

            # comparative active
            comp_act = probs.unsqueeze(-1) * self.comp_active.unsqueeze(0).to(
                probs.device
            )
            comp_act = comp_act.sum(dim=-2)  # sum over classes

            pw_a[w] = comp_sil[w][..., 0]
            pw_b[w] = comp_sil[w][..., 1]

        # A current speaker
        # Given only A is speaking we use the 'active' probability of B being the next speaker
        w = torch.where(a_current)
        p_a[w] = 1 - act_probs[w][..., 1]  # P_a = 1-P_b 
        p_b[w] = act_probs[w][..., 1]  # P_b
        if pw:
            pw_a[w] = 1 - comp_act[w][..., 1]
            pw_b[w] = comp_act[w][..., 1]

        # B current speaker
        w = torch.where(b_current)
        p_a[w] = act_probs[w][..., 0] # P_a for A being next speaker, while B is active
        p_b[w] = 1 - act_probs[w][..., 0] # P_b = 1-P_a
        if pw:
            pw_a[w] = comp_act[w][..., 0]
            pw_b[w] = 1 - comp_act[w][..., 0]

        # Both
        # P_a_prior=A is next (active)
        # P_b_prior=B is next (active)
        # We the compare/renormalize given the two values of A/B is the next speaker
        # sum = P_a_prior+P_b_prior
        # P_a = P_a_prior / sum
        # P_b = P_b_prior / sum
        w = torch.where(both)
        # Re-Normalize and compare next-active
        sum = act_probs[w][..., 0] + act_probs[w][..., 1]

        p_a[w] = act_probs[w][..., 0] / sum
        p_b[w] = act_probs[w][..., 1] / sum

        pw_probs = None
        if pw:
            sum = comp_act[w][..., 0] + comp_act[w][..., 1]
            pw_a[w] = comp_act[w][..., 0] / sum
            pw_b[w] = 1 - comp_act[w][..., 1] / sum

            pw_probs = torch.stack((pw_a, pw_b), dim=-1)
        p_probs = torch.stack((p_a, p_b), dim=-1)
        return p_probs, pw_probs

    def get_probs(self, logits, vad):
        probs = logits.softmax(dim=-1)
        next_probs, next_pw_probs = self.get_next_speaker_probs(
            probs=probs, vad=vad, pw=True
        )

        # Prediction
        ap = probs[..., self.bc_prediction[0]].sum(-1)
        bp = probs[..., self.bc_prediction[1]].sum(-1)
        bc_prediction = torch.stack((ap, bp), dim=-1)
        return {"p": next_probs, "pw": next_pw_probs, "bc_prediction": bc_prediction}

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
    from vad_turn_taking.plot_utils import plot_all_projection_windows, plot_template
    from vad_turn_taking.events import TurnTakingEvents
    from conv_ssl.evaluation.utils import load_dm

    dm = load_dm()
    diter = iter(dm.val_dataloader())

    bin_times = [0.2, 0.4, 0.6, 0.8]
    vad_hz = 100
    event_kwargs = {
        "event_pre": 0.5,
        "event_min_context": 1.0,
        "event_min_duration": 0.15,
        "event_horizon": 2.0,
        "event_start_pad": 0.05,
        "event_target_duration": 0.10,
        "event_bc_pre_silence": 1,
        "event_bc_post_silence": 2,
        "event_bc_max_active": 1,
        "event_bc_prediction_window": 0.5,
    }
    codebook = ProjectionCodebook(bin_times=bin_times)
    labeler = VadLabel(bin_times=bin_times, vad_hz=vad_hz)
    eventer = TurnTakingEvents(
        bc_idx=codebook.bc_prediction,
        horizon=event_kwargs["event_horizon"],
        min_context=event_kwargs["event_min_context"],
        start_pad=event_kwargs["event_start_pad"],
        target_duration=event_kwargs["event_target_duration"],
        pre_active=event_kwargs["event_pre"],
        bc_pre_silence=event_kwargs["event_bc_pre_silence"],
        bc_post_silence=event_kwargs["event_bc_post_silence"],
        bc_max_active=event_kwargs["event_bc_max_active"],
        bc_prediction_window=event_kwargs["event_bc_prediction_window"],
        frame_hz=vad_hz,
    )
    print("bc: ", codebook.bc_prediction.shape)
    print("shift silent: ", codebook.on_silent_shift.shape)
    print("shift active: ", codebook.on_active_shift.shape)
    print("comp silent: ", codebook.comp_silent.shape)
    print("comp active: ", codebook.comp_active.shape)

    shift_silence = codebook.idx_to_onehot(codebook.on_silent_shift[0])
    _ = plot_all_projection_windows(shift_silence)

    shift_active = codebook.idx_to_onehot(codebook.on_active_shift[0])
    _ = plot_all_projection_windows(shift_active)

    hold_active = codebook.idx_to_onehot(codebook.on_active_hold[0])
    _ = plot_all_projection_windows(hold_active)

    bc_pred_states = codebook.idx_to_onehot(codebook.bc_prediction[0])
    _ = plot_all_projection_windows(bc_pred_states)

    # Template figures
    # BC-Prediction
    fig, ax = plot_template(projection_type="bc_prediction", prefix_type="silence")
    fig, ax = plot_template(projection_type="bc_prediction", prefix_type="active")
    # Turn-shift / BC-Ongoing
    fig, ax = plot_template(projection_type="shift", prefix_type="silence")
    fig, ax = plot_template(projection_type="shift", prefix_type="active")
    fig, ax = plot_template(projection_type="shift", prefix_type="overlap")
    plt.pause(0.01)
