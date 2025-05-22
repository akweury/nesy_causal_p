
# =============================
# Train: train_context_proximity.py
# =============================
# train_context_proximity.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from context_proximity_dataset import ContextProximityDataset, context_collate_fn
from context_proximity_scorer import ContextProximityScorer
from mbg.scorer import scorer_config
EPOCHS = 10
BATCH_SIZE = 1
LR = 1e-3

model = ContextProximityScorer()
dataset = ContextProximityDataset(scorer_config.PAIR_PATH)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=context_collate_fn)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss, correct, total = 0, 0, 0
    for ci, cj, ctx, label in tqdm(data_loader):
        ci = ci
        cj = cj
        label = label

        logits = model(ci, cj, ctx[0].unsqueeze(0))
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(label)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == label).sum().item()
        total += len(label)

    print(f"[Epoch {epoch+1}] Loss: {total_loss / total:.4f} | Acc: {correct / total:.4f}")

torch.save(model.state_dict(), scorer_config.PROXIMITY_MODEL)
print(f"Model saved to {scorer_config.PROXIMITY_MODEL}")
