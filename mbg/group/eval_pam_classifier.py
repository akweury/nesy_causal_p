# Created by MacBook Pro at 23.04.25

import torch


import mbg.mbg_config as param
from mbg import patch_preprocess
from mbg.object import patch_classifier_model
from mbg.group import pam_patchset_dataset

def load_model(device):
    model_path = param.MODEL_SAVE_PATH
    model = patch_classifier_model.init_patch_classifier(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    return model




def evaluate_pam_classifier(model, data):
    device = torch.device(param.DEVICE)
    patch_sets = patch_preprocess.img_path2one_patches(data["image_path"][0])

    # label is acquired from gt.json file


    predictions = []
    for o_i in range(len(patch_sets)):
        patch_tensor = patch_sets[o_i][0][0].unsqueeze(0).to(device)  # (1, P, L, 2)
        with torch.no_grad():
            logits = model(patch_tensor)
            pred_label = logits.argmax(dim=1).item()
            predictions.append(pred_label)

    correct = (torch.tensor(predictions) == torch.tensor(labels)).sum()

    acc = correct / len(labels)
    results = [param.LABEL_NAMES[p] for p in predictions]
    return results, acc