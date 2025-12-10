import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from torch.optim import Adam
from tqdm import tqdm
import datetime

class STLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=30):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h)



if __name__ == "__main__":
    # 1. 테스트용 하이퍼파라미터 설정
    batch_size = 4       # 배차 크기
    seq_len = 60          # 입력 시퀀스 길이 (예: 과거 60분)
    input_dim = 12        # 입력 피처 개수 (Target OD + Neighbors + Inflow/Outflow + Time enc 등)
    hidden_dim = 64       # LSTM 은닉층 차원
    num_layers = 2        # LSTM 레이어 수
    output_dim = 30       # 출력 차원 (예: 미래 30분 예측)

    print("=== STLSTM Model Test Start ===")

    # 2. 모델 인스턴스 생성
    model = STLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim
    )
    
    # 모델 구조 출력
    print(model)

    # 3. 더미 입력 데이터 생성 (Random Tensor)
    # Shape: (Batch_Size, Sequence_Length, Input_Dimension)
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    print(f"\n[Input] Dummy Data Shape: {dummy_input.shape}")

    # 4. 순전파 (Forward Pass) 테스트
    try:
        prediction = model(dummy_input)
        
        print(f"[Output] Prediction Shape: {prediction.shape}")
        
        # 5. 결과 검증
        expected_shape = (batch_size, output_dim)
        if prediction.shape == expected_shape:
            print(">> Test Passed: 출력 크기가 예상과 일치합니다.")
        else:
            print(f">> Test Failed: 예상 크기 {expected_shape}와 다릅니다.")
            
    except Exception as e:
        print(f">> Error 발생: {e}")