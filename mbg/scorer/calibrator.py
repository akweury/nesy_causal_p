# Created by MacBook Pro at 03.06.25

import torch
import torch.nn as nn
import torch.optim as optim


class ConfidenceCalibrator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def train_from_data(self, X, y, device, lr=0.01, epochs=100, verbose=True, ):
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

        opt = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = torch.nn.BCELoss()

        loss_history = []

        for epoch in range(epochs):
            opt.zero_grad()
            pred = self.forward(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()

            loss_history.append(loss.item())

            # if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            #     print(f"[Epoch {epoch + 1:3d}] Loss: {loss.item():.4f}")

        # Final diagnostic
        # if verbose:
        #     print(f"\nInitial loss: {loss_history[0]:.4f} | Final loss: {loss_history[-1]:.4f}")
        #     if loss_history[-1] > loss_history[0]:
        #         print("⚠️ Loss increased — potential overfitting or bad learning rate.")
        #     elif abs(loss_history[-1] - loss_history[0]) < 1e-3:
        #         print("⚠️ Loss did not change much — may be underfitting.")
        #     else:
        #         print("✅ Loss decreased — model likely learned meaningful signal.")
