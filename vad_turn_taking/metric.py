import torch
from torchmetrics import Metric

from vad_turn_taking import time_to_frames


class ShiftHoldMetric(Metric):
    """Used in conjuction with 'VadProjection' from datasets_turntaking"""

    def __init__(
        self,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        frame_hz=100,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.frame_horizon = time_to_frames(horizon, frame_hz)
        self.frame_min_context = time_to_frames(min_context, frame_hz)
        self.frame_start_pad = time_to_frames(start_pad, frame_hz)
        self.frame_target_duration = time_to_frames(target_duration, frame_hz)
        self.frame_min_duration = self.frame_start_pad + self.frame_target_duration
        self.pred_threshold = 0.5

    def stats(self, ac, an, bc, bn):
        """
        F1 statistics over Shift/Hold

        Example 'Shift':
            * ac = shift_correct
            * an = shift_total
            * bc = hold_correct
            * bn = hold_total

        True Positives:  shift_correct
        False Negatives:  All HOLD predictions at SHIFT locations -> (shift_total - shift_correct)
        True Negatives:  All HOLD predictions at HOLD locations -> hold_correct
        False Positives:  All SHIFT predictions at HOLD locations -> (hold_total - hold_correct)

        Symmetrically true for Holds.
        """
        EPS = 1e-9
        tp = ac
        fn = an - ac
        tn = bc
        fp = bn - bc
        precision = tp / (tp + fp + EPS)
        recall = tp / (tp + fn + EPS)
        support = tp + fn
        f1 = tp / (tp + 0.5 * (fp + fn) + EPS)
        return {
            "f1": f1,
            "support": support,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    def compute(self):
        """Compute final result"""
        stats = {
            "hold": self.stats(
                ac=self.hold_correct,
                an=self.hold_total,
                bc=self.shift_correct,
                bn=self.shift_total,
            ),
            "shift": self.stats(
                ac=self.shift_correct,
                an=self.shift_total,
                bc=self.hold_correct,
                bn=self.hold_total,
            ),
        }

        # Weighted F1 score
        # scaled/weighted by the support of each metric
        # shift_f1*shift_support + hold_f1*hold_support )/ (shift_support + hold_support)
        f1h = stats["hold"]["f1"] * stats["hold"]["support"]
        f1s = stats["shift"]["f1"] * stats["shift"]["support"]
        tot = stats["hold"]["support"] + stats["shift"]["support"]
        stats["f1_weighted"] = (f1h + f1s) / tot
        return stats

    def extract_acc(self, p_next, shift, hold):
        ret = {
            "shift": {"correct": 0, "n": 0},
            "hold": {"correct": 0, "n": 0},
        }
        # shifts
        next_speaker = 0
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sa = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sa
            ret["shift"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(shift[..., next_speaker])
        if len(w[0]) > 0:
            sb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["shift"]["correct"] += sb
            ret["shift"]["n"] += len(w[0])
        # holds
        next_speaker = 0
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            ha = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += ha
            ret["hold"]["n"] += len(w[0])
        next_speaker = 1
        w = torch.where(hold[..., next_speaker])
        if len(w[0]) > 0:
            hb = (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            ret["hold"]["correct"] += hb
            ret["hold"]["n"] += len(w[0])
        return ret

    def stats_update(
        self,
        hold_correct,
        hold_total,
        shift_correct,
        shift_total,
    ):
        self.hold_correct += hold_correct
        self.hold_total += hold_total
        self.shift_correct += shift_correct
        self.shift_total += shift_total

    def update(self, p_next, vad):
        # Find valid event-frames
        hold, shift = DialogEvents.on_silence(
            vad,
            start_pad=self.frame_start_pad,
            target_frames=self.frame_target_duration,
            horizon=self.frame_horizon,
            min_context=self.frame_min_context,
            min_duration=self.frame_min_duration,
        )

        # extract TP, FP, TN, FN
        m = self.extract_acc(p_next, shift, hold)

        self.stats_update(
            hold_correct=m["hold"]["correct"],
            hold_total=m["hold"]["n"],
            shift_correct=m["shift"]["correct"],
            shift_total=m["shift"]["n"],
        )
