import torch
from vap_turn_taking.vad import VAD
from vap_turn_taking.utils import find_island_idx_len, find_label_match


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

                    # sometimes the "backchannel" consists of two smaller segments of activity
                    # and the prediction area is close to the start of both
                    # if this happens we simply fill until the first activity.
                    # That is, we do not want the area for predicting a backchannel to
                    # contain parts of the actual 'backchannel'
                    to_frame = ind[w][0]  # take first for the longest fill

                    # Omit if event occurs after datasegment
                    # VAD information is longer than the segment the
                    # model "sees" because of prediction horizon.
                    to_frame = to_frame[to_frame < max_frame]

                    if len(to_frame) < 1:
                        continue

                    # Fill region with ones
                    filler = torch.ones(
                        (to_frame - end_bc,),
                        dtype=torch.long,
                        device=bc_pred.device,
                    )
                    bc_pred[n_batch, end_bc:to_frame, speaker] = filler
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
    assert (
        prediction_window > 0
    ), f"Prediction window must be a positive integer != {prediction_window}"
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
                    start = end - prediction_window
                    if start < 0:
                        start = 0
                    valid_bc_pred[n_batch, start:end, n_speaker] = 1.0
    return valid_bc_pred


def backchannel_prediction_events(
    projection_idx,
    vad,
    bc_speaker_idx,
    prediction_window=50,
    isolated=None,
    iso_kwargs=dict(
        pre_silence_frames=100, post_silence_frames=150, max_active_frames=100
    ),
):
    # 1. Match vad with labels `projection_idx` and combine.
    bc_a = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[0])
    bc_b = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[1])

    # Backchanneler can't be last speaker
    last_speaker = VAD.get_last_speaker(vad)[..., : projection_idx.shape[1]]
    a_last = last_speaker == 0
    b_last = last_speaker == 1
    bc_a = torch.logical_and(bc_a, b_last) * 1.0
    bc_b = torch.logical_and(bc_b, a_last) * 1.0
    bc_cand = torch.stack((bc_a, bc_b), dim=-1)

    # 2. Fill prediction events until the actual backchannel starts
    bc_cand = fill_until_bc_activity(bc_cand, vad, max_fill=100)

    # 3. find as-real-as-prossible-backchannels based on isolation
    if isolated is None:
        isolated = find_isolated_activity_on_other_active(vad, **iso_kwargs)

    # 4. Match the isolated chunks with the isolated backchannel-prediction-candidate segments
    bc_pred = match_bc_pred_with_isolated(
        bc_cand, isolated, prediction_window=prediction_window
    )
    return bc_pred


######################################################################
################# Backchannels #######################################
######################################################################
def recover_bc_ongoing_negatives_single(
    vad_ongoing,
    x_ongoing,
    x_prefix,
    min_non_bc_activity,
    min_non_bc_prefix_activity,
    min_non_bc_consecutive_activity,
    min_context_frames,
    total_frames,
):
    """
    x_ongoing
    x_prefix

    min_non_bc_activity: minimal amount of activity (in frames) in order to count as a NON-backchannel
    min_non_bc_prefix_activity: minimal amount of activity, of the other speaker, prior to a NON-backchannel

    we want the negatives for backchannels to be the onset of "longer" speech which occurs after a segment, or prefix, where
    the other speaker has held the turn for some time.

    We checkviable NON-bc candidates by:

    1. "longer" activity for the current speaker (but can be separated by silence).
    2. "longer" prefix-segments where the other speaker is active, leading into the NON-bc
    """
    non_bc_cands = []

    # Find longer duration where the backchanneler or both are active
    # in other words, where it is not a backchannel but the chosen "backchanneler"
    # continues for longer. Starts of turns etc
    so, do, vo = find_island_idx_len(x_ongoing)
    so = so[vo == 1]
    do = do[vo == 1]
    if len(do) > 0:
        # only longer events from
        # w = do >= bc_neg_dur_frames
        w = do >= min_non_bc_activity
        if w.sum() > 0:
            do = do[w]
            so = so[w]
            for ss in so:
                # check that the minimal amount of consecutive information is present
                # during the NON-bc.
                if ss >= min_context_frames and ss < total_frames:
                    actual_active = vad_ongoing[
                        ss : ss + min_non_bc_consecutive_activity
                    ].sum()
                    if actual_active == min_non_bc_consecutive_activity:
                        non_bc_cands.append(ss)

    if len(non_bc_cands) == 0:
        return None

    # print('NON_bc_cands: ', non_bc_cands)
    # TODO: Actual activity longer than bc_ongoing length

    # Prefix conditions
    # Find moments where the other person, not the chosen "backchaneller", is active,
    # over a certain perdiod, which constitutes good prefix-segments for negative backchannel samples
    # A backchannel often occurs when the other speaker has ben active for some time.
    prefix_cands = []
    # s, d, v = find_island_idx_len(not_this_speaker[b, :, bcer])
    s, d, v = find_island_idx_len(x_prefix)
    s = s[v == 1]
    d = d[v == 1]
    if len(d) > 0:
        # w = d >= bc_neg_min_pre_frames
        w = d >= min_non_bc_prefix_activity
        if w.sum() > 0:
            d = d[w]
            s = s[w]
            e = s + d
            prefix_cands += e.tolist()

    if len(prefix_cands) == 0:
        return None

    # print('Prefix cand: ', prefix_cands)

    bc_negatives = []
    for bcer in non_bc_cands:
        if bcer in prefix_cands:
            bc_negatives.append(bcer)

    if len(bc_negatives) == 0:
        return None

    return bc_negatives


def recover_bc_ongoing_negatives(
    vad,
    min_context_frames=100,
    min_non_bc_activity=200,
    min_non_bc_prefix_activity=200,
    min_non_bc_consecutive_activity=50,
    total_frames=1000,
):
    ds = VAD.vad_to_dialog_vad_states(vad)
    last_speaker = VAD.get_last_speaker(vad, ds=ds)
    not_a = (last_speaker != 0) * 1.0
    not_b = (last_speaker != 1) * 1.0
    not_this_speaker = torch.stack((not_a, not_b), dim=-1)

    N = vad.shape[1]
    negatives = torch.zeros_like(vad[:, :total_frames])
    for b in range(vad.shape[0]):
        a_tmp_bcs = recover_bc_ongoing_negatives_single(
            vad_ongoing=vad[b, :, 0],
            x_ongoing=not_this_speaker[b, :, 1],
            x_prefix=not_this_speaker[b, :, 0],
            min_non_bc_activity=min_non_bc_activity,
            min_non_bc_prefix_activity=min_non_bc_prefix_activity,
            min_non_bc_consecutive_activity=min_non_bc_consecutive_activity,
            min_context_frames=min_context_frames,
            total_frames=total_frames,
        )
        if a_tmp_bcs is not None:
            for start in a_tmp_bcs:
                end = start + min_non_bc_consecutive_activity
                if min_context_frames <= start and end < N:
                    negatives[b, start:end, 0] = 1.0
        # B is backchanneler
        b_tmp_bcs = recover_bc_ongoing_negatives_single(
            vad_ongoing=vad[b, :, 1],
            x_ongoing=not_this_speaker[b, :, 0],
            x_prefix=not_this_speaker[b, :, 1],
            min_non_bc_activity=min_non_bc_activity,
            min_non_bc_prefix_activity=min_non_bc_prefix_activity,
            min_non_bc_consecutive_activity=min_non_bc_consecutive_activity,
            min_context_frames=min_context_frames,
            total_frames=total_frames,
        )
        if b_tmp_bcs is not None:
            for start in b_tmp_bcs:
                end = start + min_non_bc_consecutive_activity
                if min_context_frames <= start and end < N:
                    negatives[b, start:end, 1] = 1.0

    # if negatives.sum() == 0:
    #     return None

    return negatives


def isolated_to_bc_ongoing_positives(isolated, bc_test_frames, n_frames=1000):
    bc_pos = torch.zeros((isolated.shape[0], n_frames, 2), device=isolated.device)
    for b, bc_tmp in enumerate(isolated):
        for backchanneler in [0, 1]:
            s, d, v = find_island_idx_len(bc_tmp[..., backchanneler])
            s = s[v == 1]
            d = d[v == 1]
            # Over certain duration
            w = d >= bc_test_frames
            if len(w) > 0:
                s = s[w]
                d = d[w]
                for ss in s:
                    end = ss + bc_test_frames
                    if end <= n_frames:
                        bc_pos[b, ss:end, backchanneler] = 1.0
    return bc_pos


def find_backchannel_ongoing(
    vad,
    n_test_frames,
    pre_silence_frames,
    post_silence_frames,
    max_active_frames,
    neg_active_frames,
    neg_prefix_frames,
    min_context_frames=100,
    n_frames=1000,
):
    """
    Find backchannel (ongoing) positives/negatives
    """
    # Speaker independent activity isolation finder
    # short active segments surrounded by
    # prefix_frames (silence) | active <= max_duration_frames | Suffix frames (silence)
    isolated = find_isolated_within(
        vad,
        prefix_frames=pre_silence_frames,
        max_duration_frames=max_active_frames,
        suffix_frames=post_silence_frames,
    )
    # bc must be over 'bc_test_frames' but under 'bc_max_active_frames'
    # grab only the first 'bc_test_frames' as the actual event
    bc_pos = isolated_to_bc_ongoing_positives(
        isolated,
        bc_test_frames=n_test_frames,
        n_frames=n_frames,
    )

    bc_neg = recover_bc_ongoing_negatives(
        vad,
        min_context_frames=min_context_frames,
        min_non_bc_activity=neg_active_frames,
        min_non_bc_prefix_activity=neg_prefix_frames,  # quiet time pre NON-bc (for backchanneler)
        min_non_bc_consecutive_activity=n_test_frames,  # amount of frames where speech must be included (test area)
    )

    # Remove possible positive matches from the negative samples
    w = torch.where(bc_pos)
    if len(w[0]) > 0:
        bc_neg[w] = torch.zeros_like(w[0], dtype=torch.float)
    return bc_pos, bc_neg


######################################################################
################# Backchannels Prediction ############################
######################################################################


def recover_bc_prediction_negatives(bc_neg, neg_bc_prediction_window):
    """
    Uses the bc-ongoing negatives as reference and selects a segment of length
    neg_bc_prediction_window frames prior to the negative event.
    """
    bc_pred_neg = torch.zeros_like(bc_neg)
    for b, tmp_neg in enumerate(bc_neg):
        for backchanneler in [0, 1]:
            starts, _, v = find_island_idx_len(tmp_neg[..., backchanneler])
            starts = starts[v == 1]
            if len(starts) > 0:
                for s in starts:
                    ps = s - neg_bc_prediction_window
                    if ps > 0:
                        bc_pred_neg[b, ps:s, backchanneler] = 1.0
    return bc_pred_neg


def find_bc_prediction_negatives(vad, projection_window, ipu_lims):
    def get_cand_ipu(s, d):
        longer = d >= ipu_lims[0]
        if longer.sum() == 0:
            return None, None

        d = d[longer]
        s = s[longer]
        shorter = d <= ipu_lims[1]
        if shorter.sum() == 0:
            return None, None

        d = d[shorter]
        s = s[shorter]
        return s, d

    ds = VAD.vad_to_dialog_vad_states(vad)
    only_a = (ds == 0) * 1.0
    only_b = (ds == 3) * 1.0

    other_a = torch.logical_or(only_b, ds == 2) * 1.0
    other_b = torch.logical_or(only_a, ds == 2) * 1.0

    negs = torch.zeros_like(vad)

    for b in range(vad.shape[0]):
        break

        s1, d1, v1 = find_island_idx_len(only_a[b])
        s1 = s1[v1 == 1]
        d1 = d1[v1 == 1]
        e1 = s1 + d1
        s1_cand, d1_cand = get_cand_ipu(s1, d1)
        if s1_cand is not None:
            so, do, vo = find_island_idx_len(other_a[b])
            so = so[vo == 1]
            do = do[vo == 1]
            eo = so + do

        s2, d2, v2 = find_island_idx_len(only_b[b])
        s2 = s2[v2 == 1]
        d2 = d2[v2 == 1]
        s2_cand, d2_cand = get_cand_ipu(s2, d2)
        e2_cand = s2_cand + d2_cand

        if s2_cand is not None:
            so, do, vo = find_island_idx_len(other_b[b])
            so = so[vo == 1]
            do = do[vo == 1]
            eo = so + do

            cands2 = []
            for s_cand, d_cand in zip(s2_cand, d2_cand):

                for sother, eother in zip(so, eo):

                    if s_cand < sother and e_cand < sother:
                        e_cand = s_cand + d_cand
                        if e_cand - projection_window > 0:
                            negs[b, e_cand - projection_window : e_cand, 1] = 1.0
                    elif (
                        sother < s_cand and seother < e_cand < sother
                    ):  # candidate before other
                        if e - projection_window > 0:
                            negs[b, e - projection_window : e, 1] = 1.0

                    # sdiff = sother - s
                    # if sdiff < 0:  # other before cand
                    #     ediff = eother - s
                    #     if ediff < 0: # other end before cand
                    #         pass
                    # else: # cand before other
                    #     pass

                    # if ediff
        # for start, dur in zip(s, d):
        #     end = start+dur
        #     start = end - projection_window
        #     if start > 0:
        #         negs[b, start:end] = 1.
    return negs


######################################################################
################# Extract Probabilites from model output #############
######################################################################
def extract_backchannel_probs(p, bc_pos, bc_neg):
    """ """

    # pos_probs = []
    # neg_probs = []
    # pos_labels = []
    # neg_labels = []

    probs = []
    labels = []
    for bc_speaker in [0, 1]:
        wp = torch.where(bc_pos[..., bc_speaker])
        if len(wp[0]) > 0:
            # we know that 'bc_speaker' have a short backchannel
            # and so the given activity should  be close to zero.
            # we reverse this to get a 'regular' accuracy where probabilities
            # closer to one is considered correct

            # Onset of backchaneller results in a backchannel so we should guess
            # Next speaker != backchanneler
            pp = 1 - p[wp][..., bc_speaker]
            # pos_probs.append(pp)
            # pos_labels.append(torch.ones_like(pp))
            probs.append(pp)
            labels.append(torch.ones_like(pp, dtype=torch.long))

        wn = torch.where(bc_neg[..., bc_speaker])
        if len(wn[0]) > 0:
            # Onset of backchaneller is now something longer than a backchannel
            # i.e. a negative backchannel
            # here the model should guess
            # Next speaker == backchanneler
            # which means one minus probability(backchanneler is next speaker) -> 0
            pn = 1 - p[wn][..., bc_speaker]
            # neg_probs.append(pn)
            # neg_labels.append(torch.zeros_like(pn))
            probs.append(pn)
            labels.append(torch.zeros_like(pn, dtype=torch.long))

    # return None if no event existed
    if len(probs) > 0:
        probs = torch.cat(probs)
        labels = torch.cat(labels)
    else:
        probs = None
        labels = None
    return probs, labels


def extract_backchannel_prediction_probs(p, bc_pred_pos, bc_pred_neg):
    probs = []
    labels = []
    for next_speaker in [0, 1]:
        wp = torch.where(bc_pred_pos[..., next_speaker])
        if len(wp[0]) > 0:
            pp = p[wp][..., next_speaker]
            probs.append(pp)
            labels.append(torch.ones_like(pp, dtype=torch.long))

        wn = torch.where(bc_pred_neg[..., next_speaker])
        if len(wn[0]) > 0:
            pn = p[wn][..., next_speaker]
            probs.append(pn)
            labels.append(torch.zeros_like(pn, dtype=torch.long))

    # return None if no event existed
    if len(probs) > 0:
        probs = torch.cat(probs)
        labels = torch.cat(labels)
    else:
        probs = None
        labels = None
    return probs, labels


def extract_backchannel_prediction_probs_independent(probs):
    """

    Extract the probabilities associated with

    A:   |-|-|-|_|
    B:   |.|.|.|-|

    """
    bc_pred = []
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


def bc():
    import matplotlib.pyplot as plt
    from conv_ssl.evaluation.utils import load_dm
    from vap_turn_taking.plot_utils import (
        plot_backchannel_prediction,
        plot_projection_window,
    )
    from vap_turn_taking.vad_projection import ProjectionCodebook, VadLabel

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

    # Extract segments for backchannel prediction
    # Predict upcomming Backchannel
    bc_pred = backchannel_prediction_events(
        projection_idx,
        vad,
        prediction_window=50,  # frame duration of acceptable prediction prior to bc
        bc_speaker_idx=codebook.bc_active,
        iso_kwargs=dict(
            pre_silence_frames=100,  # minimum silence frames before bc
            post_silence_frames=200,  # minimum post silence after bc
            max_active_frames=100,  # max backchannel frame duration
        ),
    )

    fig, ax = plot_backchannel_prediction(vad, bc_pred, plot=True)


if __name__ == "__main__":

    from tqdm import tqdm
    from conv_ssl.evaluation.utils import load_dm
    from vap_turn_taking.plot_utils import plot_backchannel_prediction
    from vap_turn_taking.vad_projection import ProjectionCodebook, VadLabel
    import matplotlib.pyplot as plt

    bin_times = [0.2, 0.4, 0.6, 0.8]
    vad_hz = 100
    # Codebook to extract specific class labels from onehot-representation
    codebook = ProjectionCodebook(bin_times=bin_times, frame_hz=vad_hz)
    VL = VadLabel(bin_times=bin_times, vad_hz=vad_hz)

    # Load Data
    # The only required data is VAD (onehot encoding of voice activity) e.g. (B, N_FRAMES, 2) for two speakers
    dm = load_dm()

    ###################################################
    # Batches
    ###################################################
    diter = iter(dm.val_dataloader())
    min_context_frames = 100  # general model minimal context
    bc_pre_silence_frames = 100
    bc_post_silence_frames = 200
    bc_max_active_frames = 80  # longer than this is not a positive bc
    bc_test_frames = 20  # amount of frames at start to investigate performance
    neg_bc_active_frames = 100  # actvity including pauses for NON-bc
    neg_bc_prefix_frames = 100  # actvity including pauses for NON-bc
    neg_bc_prediction_window = 50

    batch = next(diter)
    # batch = next(iter(dm.val_dataloader()))
    projection_idx = codebook(VL.vad_projection(batch["vad"]))
    vad = batch["vad"]
    ###################################################
    # Backchannels
    ###################################################
    isolated = find_isolated_within(
        vad,
        prefix_frames=bc_pre_silence_frames,
        max_duration_frames=bc_max_active_frames,
        suffix_frames=bc_post_silence_frames,
    )
    bc_pos, bc_neg = find_backchannel_ongoing(
        vad,
        n_test_frames=bc_test_frames,
        pre_silence_frames=bc_pre_silence_frames,
        post_silence_frames=bc_post_silence_frames,
        max_active_frames=bc_max_active_frames,
        neg_active_frames=neg_bc_active_frames,
        neg_prefix_frames=neg_bc_prefix_frames,
        min_context_frames=min_context_frames,
        n_frames=1000,
    )
    # Use segment prior to bc_neg for our bc-pred-negatives
    bc_pred_neg = recover_bc_prediction_negatives(bc_neg, neg_bc_prediction_window)
    bc_pred_pos = backchannel_prediction_events(
        projection_idx,
        vad,
        bc_speaker_idx=codebook.bc_prediction,
        prediction_window=50,
        isolated=None,
        iso_kwargs=dict(
            pre_silence_frames=bc_pre_silence_frames,
            post_silence_frames=bc_post_silence_frames,
            max_active_frames=bc_max_active_frames,
        ),
    )
    # p = torch.rand_like(vad[:, :1000])
    # bc_probs, bc_labels = extract_backchannel_probs(p, bc_pos, bc_neg)
    # if bc_probs is not None:
    #     print(bc_labels)
    fig, ax = plot_backchannel_prediction(
        vad, bc_pos, bc_color="g", linewidth=3, plot=False
    )
    for i, a in enumerate(ax):
        a.plot(bc_neg[i, :, 0], color="r", linewidth=3)
        a.plot(-bc_neg[i, :, 1], color="r", linewidth=3)
        a.plot(isolated[i, :, 0], color="k", linewidth=1, linestyle="dashed")
        a.plot(-isolated[i, :, 1], color="k", linewidth=1, linestyle="dashed")
        a.plot(bc_pred_pos[i, :, 0], color="g", linewidth=2, linestyle="dashed")
        a.plot(-bc_pred_pos[i, :, 1], color="g", linewidth=2, linestyle="dashed")
        a.plot(bc_pred_neg[i, :, 0], color="r", linewidth=2, linestyle="dashed")
        a.plot(-bc_pred_neg[i, :, 1], color="r", linewidth=2, linestyle="dashed")
    plt.pause(0.01)

    ###########################################################
    ## Over entire dataset
    ###########################################################
    max_batches = -1
    dloader = dm.val_dataloader()
    if max_batches > 0:
        pbar = tqdm(dloader, total=max_batches)
    else:
        pbar = tqdm(dloader)
    n_pos, n_neg = 0, 0
    for i, batch in enumerate(pbar):
        bc_pos, bc_neg = find_backchannel_ongoing(
            batch["vad"],
            n_test_frames=bc_dict["bc_test_frames"],
            pre_silence_frames=bc_dict["bc_pre_silence_frames"],
            post_silence_frames=bc_dict["bc_post_silence_frames"],
            max_active_frames=bc_dict["bc_max_active_frames"],
            neg_active_frames=bc_dict["neg_bc_active_frames"],
            neg_prefix_frames=bc_dict["neg_bc_prefix_frames"],
            min_context_frames=bc_dict["min_context_frames"],
            n_frames=1000,
        )
        np = bc_pos.sum().item()
        nn = bc_neg.sum().item()
        n_pos += np
        n_neg += nn
        if i == max_batches:
            break

    ##################################################################
    # Extract bc-prediciton-segments and analyze distribution

    def extract_bc_prediction_segments(vad, bc_prediction):
        bc_oh = bc_prediction.sum(dim=-1)

        segments = []
        for i, bc_sample in enumerate(bc_oh):
            # find all segments
            s, d, v = find_island_idx_len(bc_sample)
            s = s[v == 1]
            d = d[v == 1]
            e = s + d

            # extract and store
            for start, end in zip(s, e):
                segments.append(vad[i, start:end])
        return segments

    from tqdm import tqdm

    segments = []
    dloader = dm.test_dataloader()
    for batch in tqdm(dloader, total=len(dloader)):
        projection_idx = codebook(VL.vad_projection(batch["vad"]))
        vad = batch["vad"]
        events = eventer(vad, projection_idx)
        s = extract_bc_prediction_segments(vad, events["backchannel_prediction"])
        segments.append(s)

        bc_oh = events["backchannel_prediction"].sum(dim=-1).long()

        vad = vad[:, : projection_idx.shape[1]]

        v = vad[torch.where(bc_oh)]
        print(v.shape)

    tensor_segments = []
    for s in segments:
        if len(s) > 0:
            tensor_segments.append(torch.stack(s))
    tensor_segments = torch.cat(tensor_segments)

    a = torch.stack(segments)

    m = tensor_segments[:, :-1].mean(dim=0)

    fig, ax = plt.subplots(1, 1)
    ax.plot(m[:, 0], color="blue")
    ax.plot(m[:, 1], color="orange")
    ax.set_ylim([0, 1])
    plt.pause(0.1)

    ###########################################
    x = torch.randn((4, 100, 32))  # (B, n_samples)
    x = x.requires_grad_(True)
    print(x.grad)  # None
    y = model(x)  # (B, N_frames, D)
    # l = y[:, :50].sum()  # gradient only step 50
    l = y[2].sum()  # gradient only step 50
    l.backward()

    g = x.grad.data.abs() * 1000
