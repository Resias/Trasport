import torch
from torch import nn


def get_support_K(kernel_type: str, cheby_order: int) -> int:
    if kernel_type == "localpool":
        if cheby_order != 1:
            raise ValueError("localpool requires cheby_order=1")
        return 1
    if kernel_type in {"chebyshev", "random_walk_diffusion"}:
        return cheby_order + 1
    if kernel_type == "dual_random_walk_diffusion":
        return cheby_order * 2 + 1
    raise ValueError(
        "kernel_type must be one of "
        "[chebyshev, localpool, random_walk_diffusion, dual_random_walk_diffusion]"
    )


class AdjProcessor:
    def __init__(self, kernel_type: str, cheby_order: int):
        self.kernel_type = kernel_type
        self.cheby_order = cheby_order if kernel_type != "localpool" else 1

    def process(self, flow: torch.Tensor) -> torch.Tensor:
        if flow.dim() == 2:
            flow = flow.unsqueeze(0)
        if flow.dim() != 3:
            raise ValueError(f"flow must be (B,N,N) or (N,N), got {tuple(flow.shape)}")

        batch_list = []
        for b in range(flow.shape[0]):
            adj = flow[b]
            kernels = []

            if self.kernel_type in {"localpool", "chebyshev"}:
                adj_norm = self.symmetric_normalize(adj)
                if self.kernel_type == "localpool":
                    kernels.append(torch.eye(adj.shape[0], device=adj.device) + adj_norm)
                else:
                    laplacian = torch.eye(adj.shape[0], device=adj.device) - adj_norm
                    laplacian = self.rescale_laplacian(laplacian)
                    kernels = self.compute_chebyshev_polynomials(laplacian)

            elif self.kernel_type == "random_walk_diffusion":
                p_forward = self.random_walk_normalize(adj)
                kernels = self.compute_chebyshev_polynomials(p_forward.T)

            elif self.kernel_type == "dual_random_walk_diffusion":
                p_forward = self.random_walk_normalize(adj)
                p_backward = self.random_walk_normalize(adj.T)
                forward = self.compute_chebyshev_polynomials(p_forward.T)
                backward = self.compute_chebyshev_polynomials(p_backward.T)
                kernels = forward + backward[1:]
            else:
                raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

            batch_list.append(torch.stack(kernels, dim=0))
        return torch.stack(batch_list, dim=0)

    @staticmethod
    def random_walk_normalize(adj: torch.Tensor) -> torch.Tensor:
        d_inv = torch.pow(adj.sum(dim=1), -1)
        d_inv[torch.isinf(d_inv)] = 0.0
        return torch.diag(d_inv) @ adj

    @staticmethod
    def symmetric_normalize(adj: torch.Tensor) -> torch.Tensor:
        d_inv_sqrt = torch.pow(adj.sum(dim=1), -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat = torch.diag(d_inv_sqrt)
        return d_mat @ adj @ d_mat

    @staticmethod
    def rescale_laplacian(laplacian: torch.Tensor) -> torch.Tensor:
        try:
            eigvals = torch.linalg.eigvals(laplacian).real
            lambda_max = torch.clamp(eigvals.max(), min=1.0)
        except RuntimeError:
            lambda_max = laplacian.new_tensor(2.0)
        return (2.0 / lambda_max) * laplacian - torch.eye(
            laplacian.shape[0],
            device=laplacian.device,
            dtype=laplacian.dtype,
        )

    def compute_chebyshev_polynomials(self, x: torch.Tensor):
        t_k = []
        for k in range(self.cheby_order + 1):
            if k == 0:
                t_k.append(torch.eye(x.shape[0], device=x.device, dtype=x.dtype))
            elif k == 1:
                t_k.append(x)
            else:
                t_k.append(2 * (x @ t_k[k - 1]) - t_k[k - 2])
        return t_k


class BDGCN(nn.Module):
    def __init__(
        self,
        K: int,
        input_dim: int,
        hidden_dim: int,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        self.activation = activation() if activation is not None else None
        self.W = nn.Parameter(
            torch.empty(self.input_dim * (self.K**2), self.hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.W)
        if self.use_bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, 0.0)

    def forward(self, x: torch.Tensor, graph):
        feat_set = []
        if isinstance(graph, torch.Tensor):
            if graph.shape[-3] != self.K:
                raise ValueError(
                    f"Static graph support size mismatch: expected K={self.K}, "
                    f"got shape={tuple(graph.shape)}"
                )
            for o in range(self.K):
                for d in range(self.K):
                    mode_1 = torch.einsum("bncl,nm->bmcl", x, graph[o])
                    mode_2 = torch.einsum("bmcl,cd->bmdl", mode_1, graph[d])
                    feat_set.append(mode_2)
        elif isinstance(graph, tuple):
            if (
                len(graph) != 2
                or graph[0].shape[-3] != self.K
                or graph[1].shape[-3] != self.K
            ):
                raise ValueError(
                    f"Dynamic graph support size mismatch: expected K={self.K}, "
                    f"got origin={tuple(graph[0].shape)}, dest={tuple(graph[1].shape)}"
                )
            for o in range(self.K):
                for d in range(self.K):
                    mode_1 = torch.einsum("bncl,bnm->bmcl", x, graph[0][:, o])
                    mode_2 = torch.einsum("bmcl,bcd->bmdl", mode_1, graph[1][:, d])
                    feat_set.append(mode_2)
        else:
            raise TypeError("graph must be a Tensor or a tuple of Tensors")

        feat_2d = torch.cat(feat_set, dim=-1)
        out = torch.einsum("bmdk,kh->bmdh", feat_2d, self.W)
        if self.use_bias:
            out = out + self.b
        if self.activation is not None:
            out = self.activation(out)
        return out


class MPGCN(nn.Module):
    def __init__(
        self,
        M: int,
        K: int,
        input_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        gcn_hidden_dim: int,
        gcn_num_layers: int,
        num_nodes: int,
        use_bias: bool = True,
        activation=None,
    ):
        super().__init__()
        self.M = M
        self.K = K
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.gcn_num_layers = gcn_num_layers

        self.branch_models = nn.ModuleList()
        for _ in range(self.M):
            branch = nn.ModuleDict()
            branch["temporal"] = nn.LSTM(
                input_size=input_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                batch_first=True,
            )
            branch["spatial"] = nn.ModuleList()
            for layer_idx in range(gcn_num_layers):
                cur_in = lstm_hidden_dim if layer_idx == 0 else gcn_hidden_dim
                branch["spatial"].append(
                    BDGCN(
                        K=K,
                        input_dim=cur_in,
                        hidden_dim=gcn_hidden_dim,
                        use_bias=use_bias,
                        activation=activation,
                    )
                )
            branch["fc"] = nn.Sequential(
                nn.Linear(gcn_hidden_dim, input_dim, bias=True),
                nn.ReLU(),
            )
            self.branch_models.append(branch)

    def init_hidden_list(self, batch_size: int):
        hidden_list = []
        weight = next(self.parameters()).data
        for _ in range(self.M):
            hidden = (
                weight.new_zeros(
                    self.lstm_num_layers,
                    batch_size * (self.num_nodes**2),
                    self.lstm_hidden_dim,
                ),
                weight.new_zeros(
                    self.lstm_num_layers,
                    batch_size * (self.num_nodes**2),
                    self.lstm_hidden_dim,
                ),
            )
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, x_seq: torch.Tensor, graph_list: list):
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(-1)
        if x_seq.dim() != 5:
            raise ValueError(f"x_seq must be (B,T,N,N) or (B,T,N,N,1), got {tuple(x_seq.shape)}")
        if x_seq.shape[2] != self.num_nodes or x_seq.shape[3] != self.num_nodes:
            raise ValueError("num_nodes mismatch between input and model")
        if len(graph_list) != self.M:
            raise ValueError("graph_list length must match M")

        batch_size, seq_len, _, _, input_dim = x_seq.shape
        if input_dim != self.input_dim:
            raise ValueError("input_dim mismatch")

        hidden_list = self.init_hidden_list(batch_size)
        lstm_in = x_seq.permute(0, 2, 3, 1, 4).reshape(
            batch_size * (self.num_nodes**2),
            seq_len,
            input_dim,
        )

        branch_out = []
        for branch_idx, branch in enumerate(self.branch_models):
            lstm_out, _ = branch["temporal"](lstm_in, hidden_list[branch_idx])
            gcn_in = lstm_out[:, -1, :].reshape(
                batch_size,
                self.num_nodes,
                self.num_nodes,
                self.lstm_hidden_dim,
            )
            for spatial_layer in branch["spatial"]:
                gcn_in = spatial_layer(gcn_in, graph_list[branch_idx])
            branch_out.append(branch["fc"](gcn_in))

        ensemble_out = torch.mean(torch.stack(branch_out, dim=-1), dim=-1)
        if ensemble_out.shape[-1] == 1:
            ensemble_out = ensemble_out.squeeze(-1)
        return ensemble_out.unsqueeze(1)
