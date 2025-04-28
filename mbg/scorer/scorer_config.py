# Created by MacBook Pro at 24.04.25

import torch
import config
from mbg.scorer.context_proximity_scorer import ContextProximityScorer

# config

PAIR_PATH = config.kp_gestalt_dataset / "train"
POS_WEIGHT = 3
PROXIMITY_MODEL = config.models / "neural_proximity_model.pt"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cpu"


def load_scorer_model(principle_name):
    if principle_name == "proximity":
        model = ContextProximityScorer()
        # model = NeuralProximityScorer(patch_len=16).to(DEVICE)
        model.load_state_dict(torch.load(PROXIMITY_MODEL))
        return model
    else:
        raise ValueError
