import torch
from vad_turn_taking.utils import find_island_idx_len, find_label_match


def find_backchannel_prediction_single(projection_idx, target_idx, fill_frames=20):
    """
    Finds segments where the `target_idx` used to define a backchannel prediction
    occurs in the source sequences `projection_idx`.

    1. `find_label_match(projection_idx, target_idx)`:
        We define backchannel prediction in the "projection_idx-space", i.e. `target_idx`.
        that is a subset of idx which represents an upcomming backchannel for each speaker.

    2. Because of the fact that we 'bin' the actual activity the associated labels are not always 'continuous'.
        That is an active bin may be active in one step but not in the next but.
        Say that bin 3 is active for a speaker: [0,0,1,0]
        the steps later (in time) can get the value: [0, 0, 0, 0] because the actual activity was not
        long enough to get a majority activation in any of the bins

        Then even later we may get the activity [0, 1, 0, 0] when a majority activation is found in the second bin
        i.e.
        [0,0,1,0] -> [0, 0, 0, 0] -> [0, 1, 0, 0]

        However, for events to match a backchannel prediction we can smooth out these steps to get a single segment which
        covers both the "real" (actual moments where the source-idx match the target-idx) and smaller segments (`fill_frames`)
        between such 'discountinuouties'.

    """
    frames, midx = find_label_match(projection_idx, target_idx)

    # fill segments between actual matches of source and target idx
    # these are by default short and is dependent on the different bin-sizes... (majority class calculation etc)
    if fill_frames > 0:
        for i, batch_frames in enumerate(frames):
            val, idx, dur = torch.unique_consecutive(
                batch_frames, return_counts=True, return_inverse=True
            )
            fill = torch.where(dur <= fill_frames)[0]
            for f in fill:
                fill_idx = torch.where(idx == f)[0]
                frames[i][fill_idx] = torch.ones_like(fill_idx)
    return frames


def fill_until_bc_activity(bc_pred, vad, max_fill=20):
    """
    Given segments where the source-idx match the associated target-idx we fill the area up until the actual backchannel activity.

    pre_bc_segment -> small silence -> actual backchannel activity

    pre_bc_segment -> actual backchannel activity

    onehot vector representation where '|' is only for emphasis
    pre_bc_onehot:    [0,1,1,1 | 0,0 | 0,0]
    actual_bc_onehot: [0,0,0,0 | 0,0 | 1,1]
    becomes:          [0,1,1,1 | 1,1 | 0,0]

    1. `fill_frames`: (If above 0)
        The indices (in label-space) are associated with upcomming backchannels and may not be defined
        on all frames leading up to the actual event. Therefore we may extend the 'event' (where we look for this prediction)
        until the frames of the actual activity spoken.
    """

    max_frame = bc_pred.shape[1]

    assert (
        bc_pred.ndim == 3
    ), f"Only implemented for batched-sequences (B, N, 2) != input: {bc_pred.shape}"
    for n_batch in range(bc_pred.shape[0]):
        for speaker in [0, 1]:
            afi, afd, afv = find_island_idx_len(bc_pred[n_batch, :, speaker])
            afi = afi[afv == 1] + afd[afv == 1]
            ind, dur, val = find_island_idx_len(vad[n_batch, :, speaker])
            ind = ind[val == 1]
            for end_bc in afi:
                bc_is_prior = end_bc <= ind
                bc_is_close = ind <= (end_bc + max_fill)
                bc_is_close = torch.logical_and(bc_is_prior, bc_is_close)
                w = torch.where(bc_is_close)[0]
                if len(w) > 0:
                    to_frame = ind[w][-1]  # take last for the longest fill

                    # Omit if event occurs after datasegment
                    # VAD information is longer than the segment the
                    # model "sees" because of prediction horizon.
                    to_frame = to_frame[to_frame < max_frame]

                    if len(to_frame) < 1:
                        continue

                    try:
                        # Fill region with ones
                        filler = torch.ones(
                            (to_frame - end_bc,),
                            dtype=torch.long,
                            device=bc_pred.device,
                        )
                        bc_pred[n_batch, end_bc:to_frame, speaker] = filler
                    except:
                        print("to_frame: ", to_frame)
                        print("end_bc: ", end_bc)
                        input()
    return bc_pred


def find_isolated_activity_on_other_active(
    vad,
    pre_silence_frames,
    post_silence_frames,
    max_active_frames,
):
    """
    Finds "short" isolated activity.

    Isolation definition
    * at least `pre-silence` (pre_silence_frames) frames of silence prior activity
    * at least `post-silence` (post_silence_frames) frames of silence after activity
    * at most `max_active_frames` of activity

    VALID targets depends on the other channel
    * the other speaker must be active in
        - `pre_silence`
        - `post_silence`

    Returns (B, N, 2) where the events are organized by who the next-speaker is.
    That is if speaker A (=0) is active and B(=1) yields a 'backchannel' then A is the next speaker.

    b = batch where with above example
    N = window where the backchannel is

    bc_cands[b, n, 0] == True  => A probabilities should be higher than B

    or the opposite:
    bc_cands[b, n, 1] == True  => B probabilities should be higher than A

    then if having a models 'next_speaker_probs', a correct behaviour is to have probability > threshold for speaker A
    during the backchannel.

    i.e. A's probabilities should be the winner at these locations


    """
    bc_cands = torch.zeros_like(vad)
    # onsets = torch.zeros_like(vad)
    for b in range(vad.shape[0]):
        for sp in [0, 1]:
            # other_active = 3 if sp == 0 else 0
            other_speaker = 1 if sp == 0 else 0
            start, dur, val = find_island_idx_len(vad[b, :, sp])
            pre_candidate = None
            for i in range(len(start) - 1):
                v0, v1 = val[i], val[i + 1]

                if v0 == 0 and v1 == 1:
                    # v0: pre-silence (before possible bc)
                    # v1: activity (possible bc)
                    if dur[i] >= pre_silence_frames:
                        #######################################################
                        # Check VALID i.e. the other speaker must have activity
                        # in the pre_silence window
                        #######################################################
                        ss = start[i + 1] - pre_silence_frames
                        pre = vad[b, ss : start[i + 1], other_speaker]
                        if pre.sum() > 0:
                            pre_candidate = i + 1
                    else:
                        pre_candidate = None

                if v0 == 1 and v1 == 0:
                    # v0: activity (possible bc)
                    # v1: post-silence (after possible bc)
                    if dur[i + 1] >= post_silence_frames:
                        if pre_candidate == i:
                            if dur[i] <= max_active_frames:
                                ###########################################################
                                # check for activity in the following (second) post_horizon
                                ###########################################################
                                post = vad[
                                    b,
                                    start[i + 1] : start[i + 1] + post_silence_frames,
                                    other_speaker,
                                ]
                                if post.sum() > 0:
                                    bc_cands[
                                        b,
                                        start[i] : start[i] + dur[i],
                                        sp,
                                    ] = 1
    return bc_cands


def match_bc_pred_with_isolated(bc_pred, isolated, prediction_window=100):
    """
    We match the segments for backchannel prediction,
            1. defined by backchannel-prediction-idx
            2. combined-smooth-segmen
            3. filled until actual activity

    with actual isolated activity (considered as-true-as-possible backchannels), which consider a longer time (before/after) span,
    to distinguish 'backchannels' from turns-with-pauses.

    If `prediction_window` > 0 then we fix the window prior to the Backchannel event to the amount of frames of `prediction_window`.

    If `prediction_window` <= 0, then we use the actual segment from the `bc_pred`.
    """
    valid_bc_pred = torch.zeros_like(bc_pred)
    for n_batch in range(bc_pred.shape[0]):
        for n_speaker in [0, 1]:
            # isolated: idx, dur, val
            isi, _, isv = find_island_idx_len(isolated[n_batch, :, n_speaker])
            # all isolated starts
            isolated_starts = isi[isv == 1]

            # skip if we don't have an 'as-real-as-prossible-backchannel'
            if len(isolated_starts) == 0:
                continue

            # backchannel candidate: idx, dur, val
            bci, bcd, bcv = find_island_idx_len(bc_pred[n_batch, :, n_speaker])
            # the backchannel end activity: start + duration of segment
            bc_starts = bci[bcv == 1]
            bcd = bcd[bcv == 1]
            bc_ends = bc_starts + bcd

            # loop over all isolated
            for iso_start in isolated_starts:
                w_end_idx = torch.where(bc_ends == iso_start)[0]
                if len(w_end_idx) > 0:
                    end = bc_ends[w_end_idx]
                    if prediction_window > 0:
                        start = end - prediction_window
                        if start < 0:
                            start = 0
                    else:
                        start = bc_starts[w]
                    valid_bc_pred[n_batch, start : end + 1, n_speaker] = 1.0
    return valid_bc_pred


def backchannel_prediction_events(
    projection_idx,
    vad,
    bc_speaker_idx,
    prediction_window=100,
    isolated=None,
    iso_kwargs=dict(
        pre_silence_frames=100, post_silence_frames=150, max_active_frames=100
    ),
):
    # 1. Match vad with labels `projection_idx` and combine.
    bc_a = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[0])
    bc_b = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[1])
    bc_pred = torch.stack((bc_a, bc_b), dim=-1)

    # 2. find as-real-as-prossible-backchannels based on isolation
    if isolated is None:
        isolated = find_isolated_activity_on_other_active(vad, **iso_kwargs)

    # 3. Match the isolated chunks with the isolated backchannel-prediction-candidate segments
    bc_pred = match_bc_pred_with_isolated(
        bc_pred, isolated, prediction_window=prediction_window
    )

    # 4. Fill prediction events until the actual backchannel starts
    bc_pred = fill_until_bc_activity(bc_pred, vad, max_fill=20)

    return bc_pred


@torch.no_grad()
def extract_backchannel_probs(p, backchannel_event):
    """
    Extracts the probabilities associated with the relevant segments, given by `backchannel_event`, whether the 'short'
    activation of a speaker is a 'backchannel' or if it's the start of a longer segment.

    That is if speaker A is active and speaker B initates an utterance (which we already know is short i.e. a 'backchannel')
    how much probability does the model put on speaker A to keep their turn?

    The model should provide a lower probability of B being the next speaker than on A holding their turn.

    Arguments:
        p:                  torch.Tensor (B, N, 2), with probabilties associated with a bc prediction from speaker 0/1 respectively
        backchannel_event : torch.Tensor (B, N, 2), one-hot encoding with segments where a backchannel prediction is valid.

    Speaker A=0:
        p [:, :, 0] -> prob of A bc prediction where, bc_event[:, :, 0] == 1
    Speaker B=1:
        p [:, :, 1] -> prob of A bc prediction where, bc_event[:, :, 1] == 1
    """

    probs = []
    for bc_speaker in [0, 1]:
        w = torch.where(backchannel_event[..., bc_speaker])
        if len(w[0]) > 0:

            # we know that 'bc_speaker' have a short backchannel
            # and so the given activity should  be close to zero.
            # we reverse this to get a 'regular' accuracy where probabilities
            # closer to one is considered correct
            probs.append(1 - p[w][..., bc_speaker])

    # return None if no event existed
    if len(probs) > 0:
        probs = torch.cat(probs)
        labels = torch.ones_like(probs, dtype=torch.long)
    else:
        probs = None
        labels = None
    return probs, labels


@torch.no_grad()
def extract_backchannel_prediction_probs(p, backchannel_event):
    """
    p:                  torch.Tensor (B, N, 2), with probabilties associated with a bc prediction from speaker 0/1 respectively
    backchannel_event : torch.Tensor (B, N, 2), one-hot encoding with segments where a backchannel prediction is valid.


    extracts the probability given by the model on defined segments `backchannel_events` were a future bc is valid.

    Speaker A=0:
        p [:, :, 0] -> prob of A bc prediction where, bc_event[:, :, 0] == 1
    Speaker B=1:
        p [:, :, 1] -> prob of A bc prediction where, bc_event[:, :, 1] == 1
    """

    probs = []
    for next_speaker in [0, 1]:
        w = torch.where(backchannel_event[..., next_speaker])
        if len(w[0]) > 0:
            probs.append(p[w][..., next_speaker])

    # return None if no event existed
    if len(probs) > 0:
        probs = torch.cat(probs)
        labels = torch.ones_like(probs, dtype=torch.long)
    else:
        probs = None
        labels = None
    return probs, labels


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from conv_ssl.evaluation.utils import load_dm
    from vad_turn_taking.plot_utils import (
        plot_backchannel_prediction,
        plot_projection_window,
    )
    from vad_turn_taking.vad_projection import ProjectionCodebook, VadLabel

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm()
    diter = iter(dm.val_dataloader())

    ####################################################################
    # Class to extract VAD-PROJECTION-WINDOWS from VAD
    VL = VadLabel(bin_times=[0.2, 0.4, 0.6, 0.8], vad_hz=100, threshold_ratio=0.5)
    # Codebook to extract specific class labels from onehot-representation
    codebook = ProjectionCodebook()

    ###################################################
    # Batch
    ###################################################
    batch = next(diter)
    vad = batch["vad"]

    # Onehot projection window. Looks into future so is not same size as original VAD
    projection_oh = VL.vad_projection(batch["vad"])
    # projection window class labels
    projection_idx = codebook(projection_oh)
    print("vad: ", tuple(vad.shape))
    print("projection_oh: ", tuple(projection_oh.shape))
    print("projection_idx: ", tuple(projection_idx.shape))

    n = bc_oh.shape[1]

    n_cols = 4
    n_rows = 7
    ii = 0
    fig, ax = plt.subplots(n_rows, n_cols, sharey=True, sharex=True)
    for row in range(n_rows):
        for col in range(n_cols):
            if ii >= n:
                break
            _ = plot_projection_window(
                proj_win=bc_oh[0, ii], bin_frames=[20, 40, 60, 80], ax=ax[row, col]
            )
            ii += 1
    plt.tight_layout()
    plt.show()

    bc_oh = codebook.idx_to_onehot(codebook.bc_active)

    # Extract segments for backchannel prediction
    # Predict upcomming Backchannel
    bc_pred = backchannel_prediction_events(
        projection_idx,
        vad,
        bc_speaker_idx=codebook.bc_active,
        iso_kwargs=dict(
            pre_silence_frames=150, post_silence_frames=150, max_active_frames=100
        ),
    )

    # Plot
    fig, ax = plot_backchannel_prediction(vad, bc_pred, plot=True)
