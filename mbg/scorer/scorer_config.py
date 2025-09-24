# Created by MacBook Pro at 24.04.25

import torch
import config
from pathlib import Path

from mbg.scorer.context_contour_scorer import ContextContourScorer
from mbg.scorer.similarity_scorer import ContextualSimilarityScorer
from mbg.scorer.slot_attention import SlotAttention  # import your saved module


# config
def get_data_path(remote, principle):
    if remote:
        data_path = Path("/gen_data") / "res_1024_pin_False" / principle
    else:
        data_path = config.grb_base / principle
    return data_path


proximity_path = config.grb_base / "proximity" / "train"
closure_path = config.grb_base / "closure" / "train"
continuity_path = config.grb_base / "continuity" / "train"
SIMILARITY_PATH = config.grb_base / "similarity" / "train"
symmetry_path = config.grb_base / "symmetry" / "train"

POS_WEIGHT = 3


def get_model_file_name(remote, principle):
    model_name = config.get_proj_output_path(remote) / f"neural_{principle}_model.pt"
    print(f"Model path: {model_name}")
    return model_name


def get_model_file_name_best(remote, principle):
    model_name = str(get_model_file_name(remote, principle)).replace(".pt", "_best.pt")
    return model_name


# PROXIMITY_MODEL = config.models / "neural_proximity_model.pt"
# SIMILARITY_MODEL = config.models / "neural_similarity_model.pt"
# CLOSURE_MODEL = config.models / "neural_closure_model.pt"
# CONTINUITY_MODEL = config.models / "neural_continuity_model.pt"
# SYMMETRY_MODEL = config.models / "neural_symmetry_model.pt"

EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cpu"


def load_scorer_model(principle_name, device, remote=False, input_dim=7):
    if principle_name == "similarity":
        input_dim = 5
    model = ContextContourScorer(input_dim=input_dim).to(device)
    model_name = get_model_file_name_best(remote, principle_name)
    model.load_state_dict(torch.load(model_name, map_location=device))
    return model
