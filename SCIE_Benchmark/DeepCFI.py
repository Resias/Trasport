import torch
import torch.nn as nn

class DeepCFI(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.counterfactual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        h = self.encoder(x)
        cf = self.counterfactual(h)   # counterfactual branch
        out = self.decoder(cf)
        return out

# model_deep_cfi = DeepCFI(input_dim=128, hidden_dim=64)
