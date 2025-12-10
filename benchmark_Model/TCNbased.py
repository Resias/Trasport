import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# ---------------------------------------------------------
# Chomp for causal trimming
# ---------------------------------------------------------
class Chomp1d(nn.Module):
    """
    Conv1d에서 causal padding으로 추가된 뒤쪽 time step을 제거.
    x: (B, C, L + padding) -> (B, C, L)
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


# ---------------------------------------------------------
# Temporal Block (Residual TCN Block)
# ---------------------------------------------------------
class TemporalBlock(nn.Module):
    """
    weight_norm Conv1d 2개 + ReLU + Dropout + Residual 연결.
    논문: filter=32, kernel_size=3, dilation=1 / 4 / (3번째는 임의 16)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        # 첫 번째 Conv
        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dp1 = nn.Dropout(dropout)

        # 두 번째 Conv
        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dp2 = nn.Dropout(dropout)

        # 채널 수가 바뀌는 경우 residual 경로를 1x1 conv로 맞춤
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) \
            if in_channels != out_channels else None

        self.out_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        return: (B, C_out, T)
        """
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
    """
    논문 TCN 모듈: 3개의 residual block.
    - Block1: 32 filters, kernel=3, dilation=1
    - Block2: 32 filters, kernel=3, dilation=4
    - Block3: 32 filters, kernel=3, dilation=16 (논문에 3번째 dilation 미명시 → 임의 설정)
    """
    def __init__(self, input_dim: int, dropout: float = 0.2):
        super().__init__()

        self.block1 = TemporalBlock(input_dim, 32, kernel_size=3, dilation=1, dropout=dropout)
        self.block2 = TemporalBlock(32, 32, kernel_size=3, dilation=4, dropout=dropout)
        self.block3 = TemporalBlock(32, 32, kernel_size=3, dilation=16, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T)
        return: (B, 32, T)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x   # (B, 32, T)


# ---------------------------------------------------------
# Dot-Product Attention
# ---------------------------------------------------------
class DotProductAttention(nn.Module):
    """
    단일 head self-attention.
    - 입력: (B, T, C)
    - 출력: (B, T, C), (B, T, T)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, C)
        return:
          out: (B, T, C)
          attn_weight: (B, T, T)
        """
        Q = self.W_q(x)        # (B, T, C)
        K = self.W_k(x)        # (B, T, C)
        V = self.W_v(x)        # (B, T, C)

        attn_score = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (B, T, T)
        attn_weight = torch.softmax(attn_score, dim=-1)                 # (B, T, T)
        out = torch.matmul(attn_weight, V)                              # (B, T, C)

        return out, attn_weight


# ---------------------------------------------------------
# TCN → Attention → LSTM → Dense
# ---------------------------------------------------------
class TCN_Attention_LSTM(nn.Module):
    """
    논문 TCN–Attention–LSTM 구조 구현.
    입력:  (B, T, F)  [예: (배치, 32 timestep, 4 변수(Y1~Y4))]
    출력:  (B, 1)     [다음 시점 OD passenger flow]
    """
    def __init__(
        self,
        input_dim: int,        # feature 수 (date, time, air quality, passenger flow 등)
        lstm_hidden: int = 128,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.0,  # 논문에서 명시 X → 기본 0
        tcn_dropout: float = 0.2
    ):
        super().__init__()

        # TCN 모듈
        self.tcn = TemporalConvNet(input_dim=input_dim, dropout=tcn_dropout)

        # Attention 모듈 (TCN 출력 채널=32)
        self.attention = DotProductAttention(dim=32)

        # LSTM 모듈
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0
        )

        # Dense layer (출력 뉴런 1개)
        self.fc = nn.Linear(lstm_hidden, 1)

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, F)
        return:
          y: (B, 1)
          att_weight: (B, T, T)
        """
        # TCN 입력 형식으로 변환: (B, F, T)
        x = x.transpose(1, 2)                 # (B, F, T)

        # TCN: 시계열 특징 추출
        tcn_out = self.tcn(x)                 # (B, 32, T)

        # Attention 입력 형식으로 변환: (B, T, C)
        tcn_out = tcn_out.transpose(1, 2)     # (B, T, 32)

        # Self-attention
        att_out, att_weight = self.attention(tcn_out)  # (B, T, 32), (B, T, T)

        # LSTM
        lstm_out, _ = self.lstm(att_out)      # (B, T, H)

        # 마지막 time step의 hidden state 사용
        last = lstm_out[:, -1, :]             # (B, H)

        # Dense → scalar
        y = self.fc(last)                     # (B, 1)

        return y, att_weight


# ---------------------------------------------------------
# main(): 실행 테스트
# ---------------------------------------------------------
def main():
    # Hyperparameters
    BATCH_SIZE = 4
    SEQ_LEN = 32       # 논문 time step = 32
    INPUT_DIM = 4      # 예: date, time, air quality, passenger flow

    model = TCN_Attention_LSTM(
        input_dim=INPUT_DIM,
        lstm_hidden=128,
        lstm_layers=3,
        lstm_dropout=0.0,   # 논문에 dropout 언급 없음
        tcn_dropout=0.2
    )

    # Dummy 데이터 생성
    dummy_x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)

    # 순전파 실행
    output, att = model(dummy_x)

    print("입력 크기 :", dummy_x.shape)   # (B, T, F)
    print("출력 크기 :", output.shape)    # (B, 1)
    print("어텐션 크기 :", att.shape)     # (B, T, T)


if __name__ == "__main__":
    main()
