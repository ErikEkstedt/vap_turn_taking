import torch
from torchmetrics import Metric

from vad_turn_taking.vad import DialogEvents
from vad_turn_taking.utils import time_to_frames


class ShiftHoldMetric(Metric):
    """Used in conjuction with 'VadProjection' from datasets_turntaking"""

    def __init__(
        self,
        horizon=1,
        min_context=1,
        start_pad=0.25,
        target_duration=0.05,
        pre_active=0.5,
        bc_pre_silence=1.5,
        bc_post_silence=3,
        bc_max_active=2,
        frame_hz=100,
        dist_sync_on_step=False,
    ):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.pred_threshold = 0.5
        self.frame_hz = frame_hz

        # Shift/Holds
        self.frame_horizon = time_to_frames(horizon, frame_hz)
        self.frame_min_context = time_to_frames(min_context, frame_hz)
        self.frame_start_pad = time_to_frames(start_pad, frame_hz)
        self.frame_target_duration = time_to_frames(target_duration, frame_hz)
        self.frame_min_duration = self.frame_start_pad + self.frame_target_duration

        # pre-active hold/shift
        self.frame_pre_active = time_to_frames(pre_active, frame_hz)

        # backchannels
        self.bc_pre_silence_frames = time_to_frames(bc_pre_silence, frame_hz)
        self.bc_post_silence_frames = time_to_frames(bc_post_silence, frame_hz)
        self.bc_max_active_frames = time_to_frames(bc_max_active, frame_hz)

        # Hold/Shift (on_silence) data
        self.add_state("hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Pre - Hold/Shift data
        self.add_state(
            "pre_hold_correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_hold_total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_shift_correct", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "pre_shift_total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )

        # Backchannels
        self.add_state("bc_correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("bc_total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def __repr__(self):
        s = "ShiftHoldMetric("
        s += f"\n  horizon: {self.frame_horizon}"
        s += f"\n  min_context: {self.frame_min_context}"
        s += f"\n  start_pad: {self.frame_start_pad}"
        s += f"\n  target_duration: {self.frame_target_duration}"
        s += f"\n  pre_active: {self.frame_pre_active}"
        s += f"\n  bc_pre_silence: {self.bc_pre_silence_frames}"
        s += f"\n  bc_post_silence: {self.bc_post_silence_frames}"
        s += f"\n  bc_max_active: {self.bc_max_active_frames}"
        s += f"\n  frame_hz: {self.frame_hz}"
        s += "\n)"
        return s

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
            "pre_hold": self.stats(
                ac=self.pre_hold_correct,
                an=self.pre_hold_total,
                bc=self.pre_shift_correct,
                bn=self.pre_shift_total,
            ),
            "pre_shift": self.stats(
                ac=self.pre_shift_correct,
                an=self.pre_shift_total,
                bc=self.pre_hold_correct,
                bn=self.pre_hold_total,
            ),
            "bc": self.bc_correct / self.bc_total,
        }

        # Weighted F1 score
        # scaled/weighted by the support of each metric
        # shift_f1*shift_support + hold_f1*hold_support )/ (shift_support + hold_support)
        f1h = stats["hold"]["f1"] * stats["hold"]["support"]
        f1s = stats["shift"]["f1"] * stats["shift"]["support"]
        tot = stats["hold"]["support"] + stats["shift"]["support"]
        stats["f1_weighted"] = (f1h + f1s) / tot

        f1h = stats["pre_hold"]["f1"] * stats["pre_hold"]["support"]
        f1s = stats["pre_shift"]["f1"] * stats["pre_shift"]["support"]
        tot = stats["pre_hold"]["support"] + stats["pre_shift"]["support"]
        stats["f1_pre_weighted"] = (f1h + f1s) / tot
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

    def extract_bc_acc(self, p_next, bc):
        ret = {"correct": 0, "n": 0}

        next_speaker = 0
        w = torch.where(bc[..., next_speaker])
        if len(w[0]) > 0:
            keep_turn = (
                (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            )
            ret["correct"] += keep_turn
            ret["n"] += len(w[0])

        next_speaker = 1
        w = torch.where(bc[..., next_speaker])
        if len(w[0]) > 0:
            keep_turn = (
                (p_next[w][..., next_speaker] > self.pred_threshold).sum().item()
            )
            ret["correct"] += keep_turn
            ret["n"] += len(w[0])
        return ret

    def extract_events(self, vad):
        hold, shift = DialogEvents.on_silence(
            vad,
            start_pad=self.frame_start_pad,
            target_frames=self.frame_target_duration,
            horizon=self.frame_horizon,
            min_context=self.frame_min_context,
            min_duration=self.frame_min_duration,
        )
        pre_hold, pre_shift = DialogEvents.get_active_pre_events(
            vad,
            hold,
            shift,
            start_pad=self.frame_start_pad,
            active_frames=self.frame_pre_active,
            min_context=self.frame_min_context,
        )
        backchannels = DialogEvents.extract_bc_candidates(
            vad,
            pre_silence_frames=self.bc_pre_silence_frames,
            post_silence_frames=self.bc_post_silence_frames,
            max_active_frames=self.bc_max_active_frames,
        )
        return {
            "hold": hold,
            "shift": shift,
            "pre_hold": pre_hold,
            "pre_shift": pre_shift,
            "backchannels": backchannels,
        }

    def event_update(self, p_next, events, bc_pre_probs=None):
        # extract TP, FP, TN, FN
        m = self.extract_acc(p_next, shift=events["shift"], hold=events["hold"])
        self.hold_correct += m["hold"]["correct"]
        self.hold_total += m["hold"]["n"]
        self.shift_correct += m["shift"]["correct"]
        self.shift_total += m["shift"]["n"]

        # Find active segment pre-events
        if bc_pre_probs is not None:
            m = self.extract_acc(
                bc_pre_probs, shift=events["pre_shift"], hold=events["pre_hold"]
            )
        else:
            m = self.extract_acc(
                p_next, shift=events["pre_shift"], hold=events["pre_hold"]
            )
        self.pre_hold_correct += m["hold"]["correct"]
        self.pre_hold_total += m["hold"]["n"]
        self.pre_shift_correct += m["shift"]["correct"]
        self.pre_shift_total += m["shift"]["n"]

        # Backchannels
        if bc_pre_probs is not None:
            m = self.extract_bc_acc(bc_pre_probs, events["backchannels"])
        else:
            m = self.extract_bc_acc(p_next, events["backchannels"])
        self.bc_correct += m["correct"]
        self.bc_total += m["n"]

    def update(self, p_next, vad, events=None, bc_pre_probs=None):
        # Find valid event-frames
        if events is None:
            events = self.extract_events(vad)
        self.event_update(p_next, events, bc_pre_probs=bc_pre_probs)
