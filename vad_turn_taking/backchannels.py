import torch

from vad_turn_taking.utils import find_island_idx_len
from vad_turn_taking.hold_shifts import get_dialog_states, get_last_speaker


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


class Backhannels:
    def __init__(
        self,
        max_duration_frames,
        pre_silence_frames,
        post_silence_frames,
        metric_dur_frames,
    ):

        assert (
            metric_dur_frames <= max_duration_frames
        ), "`metric_dur_frames` must be less than `max_duration_frames`"
        self.max_duration_frames = max_duration_frames
        self.pre_silence_frames = pre_silence_frames
        self.post_silence_frames = post_silence_frames
        self.metric_dur_frames = metric_dur_frames

    def __repr__(self):
        s = "\nBackchannel"
        s += f"\n  max_duration_frames: {self.max_duration_frames}"
        s += f"\n  pre_silence_frames: {self.pre_silence_frames}"
        s += f"\n  post_silence_frames: {self.post_silence_frames}"
        return s

    def backchannel(self, vad, last_speaker):
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
                    bc_oh[b, starts[step] : end, speaker] = 1.0
        return bc_oh

    def __call__(self, vad, last_speaker=None, ds=None):

        if ds is None:
            ds = get_dialog_states(vad)

        if last_speaker is None:
            last_speaker = get_last_speaker(vad, ds)

        bc_oh = self.backchannel(vad, last_speaker)
        return {"backchannel": bc_oh}


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from vad_turn_taking.plot_utils import plot_vad_oh

    BS = Backhannels(
        max_duration_frames=60,
        pre_silence_frames=100,
        post_silence_frames=100,
        metric_dur_frames=50,
    )
    bc = BS(vad, last_speaker)

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
