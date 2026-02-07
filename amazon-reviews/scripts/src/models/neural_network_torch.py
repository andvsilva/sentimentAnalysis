# models/neural_network_torch.py  (or inline in app.py)

import torch
import torch.nn as nn
import numpy as np

class NeuralNetworkTorch(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),

            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, X, device="cpu"):
        self.eval()

        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)

        X = X.to(device)
        probs = self(X)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict(self, X, device="cpu", threshold=0.5):
        probs = self.predict_proba(X, device=device)
        return (probs >= threshold).astype(int)
