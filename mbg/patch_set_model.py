# # Created by MacBook Pro at 17.04.25
# import torch.nn as nn
# import torch
#
#
# class PatchClassifier(nn.Module):
#     def __init__(self, input_dim=2, patch_len=16, num_patches=6, num_classes=4):
#         super().__init__()
#         self.flatten_dim = input_dim * patch_len
#         self.encoder = nn.Sequential(
#             nn.Linear(self.flatten_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU()
#         )
#         self.head = nn.Sequential(
#             nn.Linear(32 * num_patches, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_classes)
#         )
#
#     def forward(self, x):
#         # x: (B, P, 2, L)
#         B, P, C, L = x.shape
#         x = x.reshape(B, P, -1)  # (B, P, C*L)
#         x = self.encoder(x)  # (B, P, F)
#         x = x.view(B, -1)  # (B, P*F)
#         return self.head(x)
