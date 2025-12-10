# file: train_MPGCN.py
import argparse
import os
import torch
import numpy as np
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from dataset import get_mpgcn_dataset
from benchmark_Model.MPGCN import MPGCN  # 사용자가 작성한 모델 파일명에 맞게 수정
from trainer import MPGCNLM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def resolve_accelerator():
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"

def load_static_graphs(data_root, N):
    """
    Load or create static graphs (Adjacency, POI)
    Returns: adj_O, adj_D, poi_O, poi_D (torch.Tensors)
    """
    # 1. Adjacency Graph (Distance based usually)
    dist_path = os.path.join(data_root, "dist_matrix.npy")
    if os.path.exists(dist_path):
        dist_mat = np.load(dist_path)
        # Gaussian Kernel for Adjacency: exp(-dist^2 / sigma^2)
        sigma = dist_mat.std()
        adj = np.exp(- (dist_mat ** 2) / (sigma ** 2))
        adj[adj < 0.1] = 0 # Thresholding
    else:
        print("Warning: dist_matrix.npy not found. Using Random Adjacency.")
        adj = np.random.rand(N, N)
        
    adj = torch.tensor(adj, dtype=torch.float32)
    
    # 2. POI Graph (Similarity based)
    poi_path = os.path.join(data_root, "poi_similarity.npy")
    if os.path.exists(poi_path):
        poi_mat = np.load(poi_path)
    else:
        print("Warning: poi_similarity.npy not found. Using Random POI.")
        poi_mat = np.random.rand(N, N)
        
    poi = torch.tensor(poi_mat, dtype=torch.float32)
    
    # For static graphs, Origin and Dest graphs are usually same
    return adj, adj, poi, poi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/od_minute")
    parser.add_argument("--train_dir", default="train")
    parser.add_argument("--val_dir", default="test")
    
    parser.add_argument("--window_size", type=int, default=6)  # Paper uses short history
    parser.add_argument("--pred_size", type=int, default=1)    # Next step
    parser.add_argument("--hop_size", type=int, default=1)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)
    
    # Model Params
    parser.add_argument("--lstm_dim", type=int, default=32)
    parser.add_argument("--gcn_dim", type=int, default=32)
    parser.add_argument("--gcn_out", type=int, default=16)
    parser.add_argument("--cheb_k", type=int, default=3)
    
    parser.add_argument("--wandb_project", default="metro-mpgcn")
    
    args = parser.parse_args()
    DEVICE = resolve_accelerator()
    
    # 1. Dataset
    print("Loading Dataset...")
    trainset, valset = get_mpgcn_dataset(
        args.data_root, args.train_dir, args.val_dir,
        args.window_size, args.hop_size, args.pred_size
    )
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Get N from dataset
    sample = trainset[0]['x'] # (T, N, N)
    N = sample.shape[1]
    print(f"Detected Nodes N={N}")
    
    # 2. Model Init
    model = MPGCN(
        N=N, 
        lstm_h=args.lstm_dim, 
        gcn_h=args.gcn_dim, 
        gcn_out=args.gcn_out, 
        K=args.cheb_k
    )
    
    # 3. Load & Set Static Graphs
    print("Setting up Static Graphs...")
    adj_O, adj_D, poi_O, poi_D = load_static_graphs(args.data_root, N)
    
    # Move graphs to device will be handled by Lightning or manually if needed?
    # Actually, model buffers need to be on same device. 
    # Lightning handles submodule parameters but we pass tensors to `set_static_graphs`.
    # We should do this inside `on_fit_start` or pass CPU tensors and let model register them as buffers.
    # Here, for simplicity, we pass them directly. Model will calculate polys and store.
    model.set_static_graphs(adj_O, adj_D, poi_O, poi_D)
    
    # 4. Lightning System
    system = MPGCNLM(model, loss=torch.nn.MSELoss(), lr=args.lr, pred_size=args.pred_size)
    
    # 5. Trainer
    wandb_logger = WandbLogger(project=args.wandb_project)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=DEVICE,
        devices=1,
        logger=wandb_logger,
        gradient_clip_val=5.0 # RNN stability
    )
    
    print("Start Training...")
    trainer.fit(system, train_loader, val_loader)

if __name__ == "__main__":
    main()