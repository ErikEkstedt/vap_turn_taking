import torch
import vap_turn_taking.functional as VF

from typing import Dict, List, Tuple


def extract_prediction_and_targets(
    p: torch.Tensor,
    p_bc: torch.Tensor,
    events: Dict[str, List[List[Tuple[int, int, int]]]],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    batch_size = len(events["hold"])

    preds = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}
    targets = {"hs": [], "pred_shift": [], "ls": [], "pred_backchannel": []}
    for b in range(batch_size):
        ###########################################
        # Hold vs Shift
        ###########################################
        # The metrics (i.e. shift/hold) are binary so we must decide
        # which 'class' corresponds to which numeric label
        # we use Holds=0, Shifts=1
        for start, end, speaker in events["shift"][b]:
            pshift = p[b, start:end, speaker]
            preds["hs"].append(pshift)
            targets["hs"].append(torch.ones_like(pshift))
        for start, end, speaker in events["hold"][b]:
            phold = 1 - p[b, start:end, speaker]
            preds["hs"].append(phold)
            targets["hs"].append(torch.zeros_like(phold))
        ###########################################
        # Shift-prediction
        ###########################################
        for start, end, speaker in events["pred_shift"][b]:
            # prob of next speaker -> the correct next speaker i.e. a SHIFT
            pshift = p[b, start:end, speaker]
            preds["pred_shift"].append(pshift)
            targets["pred_shift"].append(torch.ones_like(pshift))
        for start, end, speaker in events["pred_shift_neg"][b]:
            # prob of next speaker -> the correct next speaker i.e. a HOLD
            phold = 1 - p[b, start:end, speaker]  # 1-shift = Hold
            preds["pred_shift"].append(phold)
            # Negatives are zero -> hold predictions
            targets["pred_shift"].append(torch.zeros_like(phold))
        ###########################################
        # Backchannel-prediction
        ###########################################
        for start, end, speaker in events["pred_backchannel"][b]:
            # prob of next speaker -> the correct next backchanneler i.e. a Backchannel
            pred_bc = p_bc[b, start:end, speaker]
            preds["pred_backchannel"].append(pred_bc)
            targets["pred_backchannel"].append(torch.ones_like(pred_bc))
        for start, end, speaker in events["pred_backchannel_neg"][b]:
            # prob of 'speaker' making a 'backchannel' in the close future
            # over these negatives this probability should be low -> 0
            # so no change of probability have to be made (only the labels are now zero)
            pred_bc = p_bc[b, start:end, speaker]  # 1-shift = Hold
            preds["pred_backchannel"].append(
                pred_bc
            )  # Negatives are zero -> hold predictions
            targets["pred_backchannel"].append(torch.zeros_like(pred_bc))
        ###########################################
        # Long vs Shoft
        ###########################################
        # TODO: Should this be the same as backchannel
        # or simply next speaker probs?
        for start, end, speaker in events["long"][b]:
            # prob of next speaker -> the correct next speaker i.e. a LONG
            plong = p[b, start:end, speaker]
            preds["ls"].append(plong)
            targets["ls"].append(torch.ones_like(plong))
        for start, end, speaker in events["short"][b]:
            # the speaker in the 'short' events is the speaker who
            # utters a short utterance: p[b, start:end, speaker] means:
            # the  speaker saying something short has this probability
            # of continue as a 'long'
            # Therefore to correctly predict a 'short' entry this probability
            # should be low -> 0
            # thus we do not have to subtract the prob from 1 (only the labels are now zero)
            # prob of next speaker -> the correct next speaker i.e. a SHORT
            pshort = p[b, start:end, speaker]  # 1-shift = Hold
            preds["ls"].append(pshort)
            # Negatives are zero -> short predictions
            targets["ls"].append(torch.zeros_like(pshort))

    # cat/stack/flatten to single tensor
    for k, v in preds.items():
        preds[k] = torch.cat(v)
    for k, v in targets.items():
        targets[k] = torch.cat(v).long()
    return preds, targets


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
