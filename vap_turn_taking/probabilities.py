import torch
import vap_turn_taking.functional as VF


class Probs:
    @staticmethod
    def marginal_probs(
        probs: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor
    ) -> torch.Tensor:
        p = []
        for next_speaker in [0, 1]:
            joint = torch.cat((pos_idx[next_speaker], neg_idx[next_speaker]), dim=-1)
            p_sum = probs[..., joint].sum(dim=-1)
            p.append(probs[..., pos_idx[next_speaker]].sum(dim=-1) / p_sum)
        return torch.stack(p, dim=-1)

    @staticmethod
    def normalize_ind_probs(probs: torch.Tensor):
        probs = probs.sum(dim=-1)  # sum all bins for each speaker
        return probs / probs.sum(dim=-1, keepdim=True)  # norm

    @staticmethod
    def silence_probs(
        p_a: torch.Tensor,
        p_b: torch.Tensor,
        sil_probs: torch.Tensor,
        silence: torch.Tensor,
    ):
        w = torch.where(silence)
        p_a[w] = sil_probs[w][..., 0]
        p_b[w] = sil_probs[w][..., 1]
        return p_a, p_b

    @staticmethod
    def single_speaker_probs(p0, p1, act_probs, current, other_speaker):
        w = torch.where(current)
        p0[w] = 1 - act_probs[w][..., other_speaker]  # P_a = 1-P_b
        p1[w] = act_probs[w][..., other_speaker]  # P_b
        return p0, p1

    @staticmethod
    def overlap_probs(p_a, p_b, act_probs, both):
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

    @staticmethod
    def next_speaker_probs(
        va: torch.Tensor, sil_probs: torch.Tensor, act_probs: torch.Tensor
    ) -> torch.Tensor:
        # Start wit all zeros
        # p_a: probability of A being next speaker (channel: 0)
        # p_b: probability of B being next speaker (channel: 1)
        p_a = torch.zeros_like(va[..., 0])
        p_b = torch.zeros_like(va[..., 0])

        # dialog states
        ds = VF.get_dialog_states(va)
        silence = ds == 1
        a_current = ds == 0
        b_current = ds == 3
        both = ds == 2

        # silence
        p_a, p_b = Probs.silence_probs(p_a, p_b, sil_probs, silence)

        # A current speaker
        # Given only A is speaking we use the 'active' probability of B being the next speaker
        p_a, p_b = Probs.single_speaker_probs(
            p_a, p_b, act_probs, a_current, other_speaker=1
        )

        # B current speaker
        # Given only B is speaking we use the 'active' probability of A being the next speaker
        p_b, p_a = Probs.single_speaker_probs(
            p_b, p_a, act_probs, b_current, other_speaker=0
        )

        # Both
        p_a, p_b = Probs.overlap_probs(p_a, p_b, act_probs, both)

        p_probs = torch.stack((p_a, p_b), dim=-1)
        return p_probs
