# Created by MacBook Pro at 27.05.25
import torch

def closure_grouping(obj_patches, group_model, threshold, device):
    feats = []
    for patch in obj_patches:
        patch_set_shifted = torch.stack(
            (patch[0][0][:, :, 0] + patch[0][1][0], patch[0][0][:, :, 1] + patch[0][1][1]), dim=2)
        feats.append(patch_set_shifted.flatten())
    feats = torch.stack(feats, dim=0)
    feats = feats.unsqueeze(0).to(device)
    _, attn = group_model(feats)  # attn: [B, N, K]
    group_ids = attn.argmax(dim=-1).squeeze(0)  # [N]

    # Convert to list-of-lists
    groups = [[] for _ in range(group_ids.max().item() + 1)]
    for idx, gid in enumerate(group_ids.tolist()):
        groups[gid].append(idx)

    # Remove empty groups (if any)
    groups = [g for g in groups if len(g) > 0]

    return groups
