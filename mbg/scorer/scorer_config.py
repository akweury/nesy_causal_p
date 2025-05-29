# Created by MacBook Pro at 24.04.25

import torch
import config
from mbg.scorer.context_contour_scorer import ContextContourScorer
from mbg.scorer.similarity_scorer import ContextualSimilarityScorer
from mbg.scorer.slot_attention import SlotAttention  # import your saved module

# config

proximity_path = config.grb_base / "proximity" / "train"
closure_path = config.grb_base / "closure" / "train"
continuity_path = config.grb_base / "continuity" / "train"
SIMILARITY_PATH = config.grb_base / "similarity" / "train"
symmetry_path = config.grb_base / "symmetry" / "train"

POS_WEIGHT = 3

PROXIMITY_MODEL = config.models / "neural_proximity_model.pt"
SIMILARITY_MODEL = config.models / "neural_similarity_model.pt"
CLOSURE_MODEL = config.models / "neural_closure_model.pt"
CONTINUITY_MODEL = config.models / "neural_continuity_model.pt"
SYMMETRY_MODEL = config.models / "neural_symmetry_model.pt"

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cpu"


def load_scorer_model(principle_name, device, input_dim=7):
    model = ContextContourScorer(input_dim=input_dim).to(device)
    if principle_name == "proximity":
        model.load_state_dict(torch.load(PROXIMITY_MODEL, map_location=device))
    elif principle_name == "similarity":
        model.load_state_dict(torch.load(SIMILARITY_MODEL, map_location=device))
    elif principle_name == "closure":
        model.load_state_dict(torch.load(CLOSURE_MODEL, map_location=device))
    elif principle_name == "continuity":
        model.load_state_dict(torch.load(CONTINUITY_MODEL, map_location=device))
    elif principle_name == "symmetry":
        model.load_state_dict(torch.load(SYMMETRY_MODEL, map_location=device))
    else:
        raise ValueError
    return model
