# Created by MacBook Pro at 03.06.25

import torch
import torch.nn as nn




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
