# Created by MacBook Pro at 27.05.25


import torch.nn.functional as F
import torch
from mbg.scorer import affinity_model
from mbg.scorer.contour_group_dataset import ContourGroupDataset, collate_group_batch
from torch.utils.data import DataLoader
from mbg.scorer import scorer_config

def group_ids_to_affinity(group_ids):
    """
    group_ids: (B, N)
    Returns: (B, N, N) binary matrix
    """
    B, N = group_ids.shape
    group_ids = group_ids.unsqueeze(2)  # [B, N, 1]
    mask_i = group_ids.expand(B, N, N)
    mask_j = group_ids.transpose(1, 2).expand(B, N, N)
    affinity = (mask_i == mask_j) & (mask_i != -1)
    return affinity.float()


def loss_fn(pred_affinity, true_affinity, valid_mask):
    # valid_mask: (B, N), 1 = valid, 0 = padded
    B, N = valid_mask.shape
    mask = (valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)).bool()  # [B, N, N]
    loss = F.binary_cross_entropy_with_logits(pred_affinity, true_affinity, reduction='none')
    return (loss * mask).sum() / mask.sum()


data_path = scorer_config.closure_path
model_path = scorer_config.CLOSURE_MODEL

dataset = ContourGroupDataset(data_path)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_group_batch)
device = "cpu"
model = affinity_model.AffinityPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    count = 0

    for patches, mask, gids in loader:
        patches = patches.to(device)
        mask = mask.to(device)
        gids = gids.to(device)

        # Forward
        aff_pred = model(patches, mask)
        aff_gt = group_ids_to_affinity(gids).to(device)
        loss = loss_fn(aff_pred, aff_gt, mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    avg_loss = total_loss / count
    print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

# Save model
save_path = "affinity_predictor_model.pt"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")