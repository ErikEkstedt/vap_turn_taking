from tqdm import tqdm
import torch

from vad_turn_taking.events import TurnTakingEvents
from vad_turn_taking.vad_projection import VadLabel, ProjectionCodebook
from conv_ssl.evaluation.utils import load_dm
from conv_ssl.utils import to_device

from vad_turn_taking.backchannel import (
    find_backchannel_prediction_single,
    fill_until_bc_activity,
    find_isolated_activity_on_other_active,
    match_bc_pred_with_isolated,
)
from vad_turn_taking.plot_utils import plot_backchannel_prediction


if __name__ == "__main__":

    dm = load_dm(batch_size=4, num_workers=4)

    labeler = VadLabel()
    codebook = ProjectionCodebook().cuda()
    eventer = TurnTakingEvents(codebook.bc_active)

    stats = {"bc_pred": 0, "bc_ongoing": 0, "total": 0}
    for batch in tqdm(dm.train_dataloader()):
        batch = to_device(batch, "cuda")
        projection_oh = labeler.vad_projection(batch["vad"])
        projection_idx = codebook(projection_oh)
        try:
            events = eventer(batch["vad"], projection_idx)
        except:

            projection_oh = labeler.vad_projection(batch["vad"])
            projection_idx = codebook(projection_oh).cpu()
            bc_speaker_idx = codebook.bc_active.cpu()
            vad = batch["vad"].cpu()
            prediction_window = 100
            iso_kwargs = dict(
                pre_silence_frames=100,  # minimum silence frames before bc
                post_silence_frames=200,  # minimum post silence after bc
                max_active_frames=100,  # max backchannel frame duration
            )

            # 1. Match vad with labels `projection_idx` and combine.
            bc_a = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[0])
            bc_b = find_backchannel_prediction_single(projection_idx, bc_speaker_idx[1])
            bc_pred = torch.stack((bc_a, bc_b), dim=-1)
            fig, ax = plot_backchannel_prediction(vad, bc_pred, plot=True)
            __import__("ipdb").set_trace()

            # 2. Fill prediction events until the actual backchannel starts
            # bc_pred = fill_until_bc_activity(bc_pred, vad, max_fill=20)
            # fig, ax = plot_backchannel_prediction(vad, bc_pred, plot=True)
            #
            # # 3. find as-real-as-prossible-backchannels based on isolation
            # # if isolated is None:
            # isolated = find_isolated_activity_on_other_active(vad, **iso_kwargs)
            #
            # # 4. Match the isolated chunks with the isolated backchannel-prediction-candidate segments
            # bc_pred = match_bc_pred_with_isolated(
            #     bc_pred, isolated, prediction_window=prediction_window
            # )
            # fig, ax = plot_backchannel_prediction(vad, bc_pred, plot=True)
            # __import__("ipdb").set_trace()
            input()

        stats["bc_pred"] += (
            (events["backchannel_prediction"].sum(-1).sum(-1) > 0).sum().item()
        )
        stats["bc_ongoing"] += (events["backchannel"].sum(-1).sum(-1) > 0).sum().item()
        stats["total"] += batch["vad"].shape[0]
