from tqdm import tqdm

from vad_turn_taking.events import TurnTakingEvents
from vad_turn_taking.vad_projection import VadLabel, ProjectionCodebook
from conv_ssl.evaluation.utils import load_dm
from conv_ssl.utils import to_device


if __name__ == "__main__":

    dm = load_dm(batch_size=10, num_workers=24)

    labeler = VadLabel()
    codebook = ProjectionCodebook().cuda()
    eventer = TurnTakingEvents(codebook.bc_active)

    stats = {"bc_pred": 0, "bc_ongoing": 0}
    for batch in tqdm(dm.train_dataloader()):
        batch = to_device(batch, "cuda")
        projection_oh = labeler.vad_projection(batch["vad"])
        projection_idx = codebook(projection_oh)
        events = eventer(batch["vad"], projection_idx)
        stats["bc_pred"] += (
            (events["backchannel_prediction"].sum(-1).sum(-1) > 0).sum().item()
        )
        stats["bc_ongoing"] += (events["backchannel"].sum(-1).sum(-1) > 0).sum().item()
