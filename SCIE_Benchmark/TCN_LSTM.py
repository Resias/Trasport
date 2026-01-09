import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size=3, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(c_in, c_out, kernel_size,
                              padding=(kernel_size-1)*dilation,
                              dilation=dilation)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.conv(x))

class TCN_LSTM_Attn(nn.Module):
    def __init__(self, n_features, tcn_channels, lstm_hidden):
        super().__init__()
        self.tcn1 = TCNBlock(n_features, tcn_channels, kernel_size=3, dilation=1)
        self.lstm = nn.LSTM(tcn_channels, lstm_hidden, batch_first=True)
        self.attn = nn.Linear(lstm_hidden, 1)
        self.fc = nn.Linear(lstm_hidden, n_features)

    def forward(self, x):
        # x: [B, seq_len, features]
        x_tcn = self.tcn1(x.transpose(1,2))  # [B, channels, seq_len]
        x_tcn = x_tcn.transpose(1,2)        # [B, seq_len, channels]
        out, _ = self.lstm(x_tcn)           # [B, seq_len, hidden]
        # Attention
        weights = torch.softmax(self.attn(out), dim=1)  # [B, seq_len, 1]
        context = (weights * out).sum(dim=1)            # [B, hidden]
        return self.fc(context)

# model_tcn_attn = TCN_LSTM_Attn(n_features=16, tcn_channels=32, lstm_hidden=64)
