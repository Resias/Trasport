import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------
# Chomp for causal trimming
# ---------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size]


# ---------------------------------------------------------
# Temporal Block
# ---------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

        self.out_relu = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.dp1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.dp2(y)

        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(y + res)


# ---------------------------------------------------------
# TCN (3 Residual Blocks per paper)
# ---------------------------------------------------------
class TemporalConvNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block1 = TemporalBlock(input_dim, 32, kernel_size=3, dilation=1)
        self.block2 = TemporalBlock(32, 32, kernel_size=3, dilation=4)
        self.block3 = TemporalBlock(32, 32, kernel_size=3, dilation=16)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x   # (B, 32, T)


# ---------------------------------------------------------
# Dot-Product Attention
# ---------------------------------------------------------
class DotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x):
        # x: (B, T, C)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        attn_weight = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(attn_weight, V)

        return out, attn_weight


# ---------------------------------------------------------
# TCN → Attention → LSTM → Dense
# ---------------------------------------------------------
class TCN_Attention_LSTM(nn.Module):
    def __init__(self, input_dim, lstm_hidden=128, lstm_layers=3):
        super().__init__()

        self.tcn = TemporalConvNet(input_dim=input_dim)
        self.attention = DotProductAttention(dim=32)

        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.transpose(1, 2)     # → (B, F, T)
        tcn_out = self.tcn(x)

        tcn_out = tcn_out.transpose(1, 2)  # → (B, T, 32)
        att_out, att_weight = self.attention(tcn_out)

        lstm_out, _ = self.lstm(att_out)

        last = lstm_out[:, -1, :]
        y = self.fc(last)

        return y, att_weight



# ---------------------------------------------------------
# main(): 실행 테스트
# ---------------------------------------------------------
def main():
    # Hyperparameters
    BATCH_SIZE = 4
    SEQ_LEN = 32       # 논문 timestep = 32
    INPUT_DIM = 4      # date, time, air quality, passenger flow

    model = TCN_Attention_LSTM(
        input_dim=INPUT_DIM,
        lstm_hidden=128,
        lstm_layers=3
    )

    # Dummy 데이터 생성
    dummy_x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # 순전파 실행
    output, att = model(dummy_x)

    print("입력 크기 :", dummy_x.shape)
    print("출력 크기 :", output.shape)   # (B, 1)
    print("어텐션 크기 :", att.shape)   # (B, T, T)


if __name__ == "__main__":
    main()
