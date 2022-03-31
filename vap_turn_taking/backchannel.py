import torch
from vap_turn_taking.utils import find_island_idx_len
from vap_turn_taking.hold_shifts import get_dialog_states, get_last_speaker


def find_isolated_within(vad, prefix_frames, max_duration_frames, suffix_frames):
    """
    ... <= prefix_frames (silence) | <= max_duration_frames (active) | <= suffix_frames (silence) ...
    """

    isolated = torch.zeros_like(vad)
    for b, vad_tmp in enumerate(vad):
        for speaker in [0, 1]:
            starts, durs, vals = find_island_idx_len(vad_tmp[..., speaker])
            for step in range(1, len(starts) - 1):
                # Activity condition: current step is active
                if vals[step] == 0:
                    continue

                # Prefix condition:
                # check that current active step comes after a certain amount of inactivity
                if durs[step - 1] < prefix_frames:
                    continue

                # Suffix condition
                # check that current active step comes after a certain amount of inactivity
                if durs[step + 1] < suffix_frames:
                    continue

                current_dur = durs[step]
                if current_dur <= max_duration_frames:
                    start = starts[step]
                    end = start + current_dur
                    isolated[b, start:end, speaker] = 1.0
    return isolated


class Backchannel:
    def __init__(
        self,
        max_duration_frames,
        min_duration_frames,
        pre_silence_frames,
        post_silence_frames,
        metric_dur_frames,
        metric_pre_label_dur,
    ):

        assert (
            metric_dur_frames <= max_duration_frames
        ), "`metric_dur_frames` must be less than `max_duration_frames`"
        self.max_duration_frames = max_duration_frames
        self.min_duration_frames = min_duration_frames
        self.pre_silence_frames = pre_silence_frames
        self.post_silence_frames = post_silence_frames
        self.metric_dur_frames = metric_dur_frames
        self.metric_pre_label_dur = metric_pre_label_dur

    def __repr__(self):
        s = "\nBackchannel"
        s += f"\n  max_duration_frames: {self.max_duration_frames}"
        s += f"\n  pre_silence_frames: {self.pre_silence_frames}"
        s += f"\n  post_silence_frames: {self.post_silence_frames}"
        return s

    def backchannel(self, vad, last_speaker, max_frame=None, min_context=0):
        """
        Finds backchannel based on VAD signal. Iterates over batches and speakers.

        Extracts segments of activity/non-activity to find backchannels.

        Backchannel Conditions

        * Backchannel activity must be shorter than `self.max_duration_frames`
        * Backchannel activity must follow activity from the other speaker
        * Silence prior to backchannel, in the "backchanneler" channel, must be greater than `self.pre_silence_frames`
        * Silence after backchannel, in the "backchanneler" channel, must be greater than `self.pre_silence_frames`
        """

        bc_oh = torch.zeros_like(vad)
        pre_bc_oh = torch.zeros_like(vad)
        for b, vad_tmp in enumerate(vad):

            for speaker in [0, 1]:
                other_speaker = 0 if speaker == 1 else 1

                starts, durs, vals = find_island_idx_len(vad_tmp[..., speaker])
                for step in range(1, len(starts) - 1):
                    # Activity condition: current step is active
                    if vals[step] == 0:
                        continue

                    # Activity duration condition: segment must be shorter than
                    # a certain number of frames
                    if durs[step] > self.max_duration_frames:
                        continue

                    if durs[step] < self.min_duration_frames:
                        continue

                    start = starts[step]

                    # Shift-ish condition:
                    # Was the other speaker active prior to this `backchannel` candidate?
                    # If not than this is a short IPU in the middle of a turn
                    pre_speaker_cond = last_speaker[b, start - 1] == other_speaker
                    if not pre_speaker_cond:
                        continue

                    # Prefix condition:
                    # check that current active step comes after a certain amount of inactivity
                    if durs[step - 1] < self.pre_silence_frames:
                        continue

                    # Suffix condition
                    # check that current active step comes after a certain amount of inactivity
                    if durs[step + 1] < self.post_silence_frames:
                        continue

                    # Add segment as a backchanel
                    end = starts[step] + durs[step]
                    if self.metric_dur_frames > 0:
                        end = starts[step] + self.metric_dur_frames

                    # Max Frame condition:
                    # can't have event outside of predictable window
                    if max_frame is not None:
                        if end >= max_frame:
                            continue

                    # Min Context condition:
                    if starts[step] < min_context:
                        continue

                    bc_oh[b, starts[step] : end, speaker] = 1.0

                    # Min Context condition:
                    if (starts[step] - self.metric_pre_label_dur) < min_context:
                        continue

                    pre_bc_oh[
                        b,
                        starts[step] - self.metric_pre_label_dur : starts[step],
                        speaker,
                    ] = 1.0
        return bc_oh, pre_bc_oh

    def __call__(self, vad, last_speaker=None, ds=None, max_frame=None, min_context=0):

        if ds is None:
            ds = get_dialog_states(vad)

        if last_speaker is None:
            last_speaker = get_last_speaker(vad, ds)

        bc_oh, pre_bc = self.backchannel(
            vad, last_speaker, max_frame=max_frame, min_context=min_context
        )
        return {"backchannel": bc_oh, "pre_backchannel": pre_bc}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from vap_turn_taking.plot_utils import plot_vad_oh

    BS = Backhannel(**bs_dict)
    tt_bc = BS(va)

    (tt_bc["backchannel"] != bc).sum()

    n_rows = 4
    n_cols = 4
    fig, ax = plt.subplots(n_rows, n_cols, sharey=True, sharex=True, figsize=(16, 4))
    b = 0
    for row in range(n_rows):
        for col in range(n_cols):
            _ = plot_vad_oh(vad[b], ax=ax[row, col])
            _ = plot_vad_oh(
                bc["backchannel"][b],
                ax=ax[row, col],
                colors=["purple", "purple"],
                alpha=0.8,
            )
            b += 1
            if b == vad.shape[0]:
                break
        if b == vad.shape[0]:
            break
    plt.pause(0.1)
