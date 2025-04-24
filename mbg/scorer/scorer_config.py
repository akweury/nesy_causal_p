# Created by MacBook Pro at 24.04.25

import config

# config

PAIR_PATH = config.kp_gestalt_dataset / "train"

PROXIMITY_MODEL = config.models / "neural_proximity_model.pt"
EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cpu"
