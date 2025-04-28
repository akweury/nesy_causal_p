# # Created by MacBook Pro at 24.04.25
# # train_proximity.py
# import wandb
# from tqdm import tqdm
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from mbg.scorer.proximity_pair_dataset import ProximityPairDataset
# from mbg.scorer.neural_proximity import NeuralProximityScorer
# from mbg.scorer import scorer_config
#
# # data
# dataset = ProximityPairDataset(scorer_config.PAIR_PATH)
# loader = DataLoader(dataset, batch_size=scorer_config.BATCH_SIZE, shuffle=True)
#
# # model
#
# # ========== 初始化 ==========
# wandb.init(project="proximity_pair_scorer")
#
# dataset = ProximityPairDataset(scorer_config.PAIR_PATH)
# dataloader = DataLoader(dataset, batch_size=scorer_config.BATCH_SIZE, shuffle=True)
#
# model = NeuralProximityScorer(patch_len=16).to(scorer_config.DEVICE)
# optimizer = optim.Adam(model.parameters(), lr=scorer_config.LR)
# criterion = nn.BCEWithLogitsLoss()
#
# # ========== 训练 ==========
# for epoch in range(scorer_config.EPOCHS):
#     model.train()
#     total_loss = 0.0
#     total_correct = 0
#     total_samples = 0
#
#     for patch_i, patch_j, label in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
#         patch_i, patch_j, label = (patch_i.to(scorer_config.DEVICE), patch_j.to(scorer_config.DEVICE),
#                                    label.to(scorer_config.DEVICE))
#
#         optimizer.zero_grad()
#         logits = model(patch_i, patch_j)
#         loss = criterion(logits, label)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item() * len(label)
#         preds = (torch.sigmoid(logits) > 0.5).float()
#         total_correct += (preds == label).sum().item()
#         total_samples += len(label)
#
#     acc = total_correct / total_samples
#     wandb.log({"loss": total_loss / total_samples, "accuracy": acc, "epoch": epoch + 1})
#     print(f"Epoch {epoch+1} | Loss: {total_loss / total_samples:.4f} | Acc: {acc:.4f}")
#
# # 保存模型
# torch.save(model.state_dict(), scorer_config.PROXIMITY_MODEL)
# print(f"Model saved to {scorer_config.PROXIMITY_MODEL}")