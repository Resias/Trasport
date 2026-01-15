# STDAMHGN.py
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperSageLayer(nn.Module):
    """
    HyperSage convolution layer (p = 1)
    """

    def __init__(self, in_dim, out_dim, Q=64):
        super().__init__()
        self.Q = Q
        self.fc = nn.Linear(in_dim, out_dim)

    def _sample_hyperedge(self, e, Q):
        if len(e) <= Q:
            return e
        return random.sample(e, Q)

    def forward(self, X, hyperedges):
        """
        X          : (B, |V|, F)
        hyperedges : List[List[int]]
        """
        B, V, F = X.shape
        device = X.device

        X_out = torch.zeros(B, V, F, device=device)

        # hyperedge-wise aggregation
        deg = torch.zeros(V, device=device)

        for e in hyperedges:
            e = self._sample_hyperedge(e, self.Q)
            e = torch.tensor(e, device=device)
            agg = X[:, e, :].mean(dim=1)
            X_out[:, e, :] += agg.unsqueeze(1)
            deg[e] += 1

        deg = deg.clamp(min=1.0)
        X_out = X_out / deg.view(1, V, 1)
        return self.fc(X_out)

class DAMHG(nn.Module):
    """
    Dynamic Attentive Multi-HyperGraph (feature extractor part)
    """

    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            HyperSageLayer(in_dim, hid_dim),
            HyperSageLayer(hid_dim, hid_dim)
        ])

    def forward(self, X, hypergraphs):
        outs = []
        for hg in hypergraphs:
            h = X
            for layer in self.layers:
                h = layer(h, hg)
            outs.append(h)
        return outs

class HypergraphAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim, bias=False)
        self.a = nn.Linear(2 * dim, 1, bias=False)

    def forward(self, Xs):
        """
        Xs : List[(B, |V|, D)]
        """
        H = len(Xs)
        B, V, D = Xs[0].shape

        # reference embedding (mean)
        ref = torch.mean(torch.stack(Xs, dim=0), dim=0)

        scores = []
        for X in Xs:
            z = F.leaky_relu(
                self.a(torch.cat([self.W(X), self.W(ref)], dim=-1)),
                negative_slope=0.2
            )                       # (B, |V|, 1)
            scores.append(z)
        # softmax over hypergraph dimension (paper Eq.11)
        alpha = torch.softmax(torch.stack(scores, dim=0), dim=0)

        Y = 0
        for i in range(H):
            Y = Y + alpha[i] * Xs[i]

        return Y    # (B, |V|, D)

class STDAMHGN(nn.Module):
    def __init__(
        self,
        num_vertices,
        m, n,
        hid_dim,
        hypergraphs
    ):
        super().__init__()

        self.V = num_vertices
        self.m = m
        self.n = n

        # fixed hypergraphs (list of 4)
        self.hypergraphs = hypergraphs

        # DAMHG
        self.damhg = DAMHG(1, hid_dim)
        self.attn = HypergraphAttention(hid_dim)

        # tendency branch
        self.lstm = nn.LSTM(
            input_size=hid_dim,
            hidden_size=hid_dim,
            batch_first=True
        )

        # fusion
        self.fc_out = nn.Linear(2 * hid_dim, 1)

    def forward(self, tendency, periodicity):
        """
        tendency    : (B, m, |V|)
        periodicity : (B, n, |V|)
        """

        B = tendency.size(0)

        # ---------- tendency branch ----------
        t_seq = []
        for k in range(self.m):
            x = tendency[:, k, :].unsqueeze(-1)   # (B, |V|, 1)
            xs = self.damhg(x, self.hypergraphs)
            x_fused = self.attn(xs)                # (B, |V|, D)
            t_seq.append(x_fused)

        # t_seq: (B, m, |V|, D)
        t_seq = torch.stack(t_seq, dim=1)   # (B, m, |V|, D)
        B, m, V, D = t_seq.shape

        # vertex-wise LSTM: (B*V, m, D)
        t_seq_v = t_seq.permute(0, 2, 1, 3).contiguous()
        t_seq_v = t_seq_v.view(B * V, m, D)

        _, (h_t, _) = self.lstm(t_seq_v)
        h_t = h_t[-1]                       # (B*V, D)

        h_t = h_t.view(B, V, D)             # (B, |V|, D)

        # ---------- periodicity branch ----------
        p_seq = []
        for k in range(self.n):
            x = periodicity[:, k, :].unsqueeze(-1)
            xs = self.damhg(x, self.hypergraphs)
            x_fused = self.attn(xs)
            p_seq.append(x_fused)

        p_seq = torch.stack(p_seq, dim=1)          # (B, n, |V|, D)
        # periodicity vertex-wise LSTM
        p_seq_v = p_seq.permute(0, 2, 1, 3).contiguous()
        p_seq_v = p_seq_v.view(B * V, self.n, D)

        _, (h_p, _) = self.lstm(p_seq_v)
        h_p = h_p[-1].view(B, V, D)


        # ---------- fusion (vertex-wise) ----------
        fusion_feat = torch.cat(
            [h_t, h_p],
            dim=-1
        )   

        out = self.fc_out(fusion_feat)       # (B, |V|, 1)
        return out.squeeze(-1)               # (B, |V|)
