from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATConv
    from torch_geometric.utils import dense_to_sparse
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
    GATConv = None  # type: ignore[assignment]
    dense_to_sparse = None  # type: ignore[assignment]
    _PYG_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _PYG_IMPORT_ERROR = None

try:
    from mamba_ssm import Mamba
except ModuleNotFoundError as exc:  # pragma: no cover - depends on runtime environment
    Mamba = None  # type: ignore[assignment]
    _MAMBA_IMPORT_ERROR: ModuleNotFoundError | None = exc
else:
    _MAMBA_IMPORT_ERROR = None


def _require_model_dependencies() -> None:
    if _PYG_IMPORT_ERROR is not None:
        raise ImportError(
            "torch-geometric is required to construct FinMamba. Install the project "
            "dependencies before training."
        ) from _PYG_IMPORT_ERROR
    if _MAMBA_IMPORT_ERROR is not None:
        raise ImportError(
            "mamba-ssm is required to construct FinMamba. Install the project "
            "dependencies before training."
        ) from _MAMBA_IMPORT_ERROR


def topk_matrix(adj_matrix: torch.Tensor, k: int) -> torch.Tensor:
    if adj_matrix.ndim != 2 or adj_matrix.size(0) != adj_matrix.size(1):
        raise ValueError(f"adj_matrix must be square, got {tuple(adj_matrix.shape)}")

    device = adj_matrix.device
    n = adj_matrix.size(0)
    triu_indices = torch.triu_indices(n, n, offset=1, device=device)
    triu_values = adj_matrix[triu_indices[0], triu_indices[1]]
    if k < 0 or k > triu_values.numel():
        raise ValueError(f"k={k} is outside [0, {triu_values.numel()}]")

    topk_values, topk_indices = torch.topk(triu_values, k)
    new_adj = torch.zeros_like(adj_matrix)

    row = triu_indices[0][topk_indices]
    col = triu_indices[1][topk_indices]
    new_adj[row, col] = topk_values
    new_adj[col, row] = topk_values
    new_adj.fill_diagonal_(1)
    return new_adj


def gib_loss(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Graph information bottleneck loss, kept mathematically identical to the source."""
    if x.shape != z.shape:
        raise ValueError(
            f"GIB loss requires x and z to share a shape, got {tuple(x.shape)} and "
            f"{tuple(z.shape)}. Set GAT out channels equal to the feature dimension."
        )
    z_mean = torch.mean(z, dim=0)
    x_mean = torch.mean(x, dim=0)
    z_var = torch.var(z, dim=0)
    x_var = torch.var(x, dim=0)
    loss = torch.sum((z_mean - x_mean) ** 2) / torch.sum(z_var + x_var)
    return torch.clamp(loss, max=0.5)


class MarketGuideInception(nn.Module):
    def __init__(
        self,
        input_dim: int,
        kernel_sizes: Sequence[int] = (4, 10, 20),
        init_sparsity: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not kernel_sizes or any(size <= 0 for size in kernel_sizes):
            raise ValueError("kernel_sizes must contain positive integers")

        self.branches = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=input_dim,
                    out_channels=1,
                    kernel_size=size,
                    padding=size // 2,
                )
                for size in kernel_sizes
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=input_dim)
        self.fc = nn.Linear(len(kernel_sizes) * input_dim, 1)
        self.dim = input_dim
        self.init_sparsity = init_sparsity

    def forward(self, market_index: torch.Tensor) -> torch.Tensor:
        market_index = market_index.unsqueeze(0).transpose(1, 2)
        branch_outputs = [self.pool(branch(market_index)) for branch in self.branches]
        combined_out = torch.cat(branch_outputs, dim=1)
        combined_out = combined_out.view(-1, len(self.branches) * self.dim)
        market_signal = self.fc(combined_out)
        return self.init_sparsity / (1 + torch.exp(-market_signal))


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        num_heads: int = 2,
    ) -> None:
        super().__init__()
        _require_model_dependencies()
        if num_layers < 2:
            raise ValueError("The legacy GAT topology requires num_layers >= 2")
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList(
            [GATConv(in_channels, hidden_channels, heads=num_heads, concat=True)]
        )
        for _ in range(num_layers - 2):
            self.gat_layers.append(
                GATConv(
                    hidden_channels * num_heads,
                    hidden_channels,
                    heads=num_heads,
                    concat=True,
                )
            )
        self.gat_layers.append(
            GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for layer_index, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            if layer_index != self.num_layers - 1:
                x = F.gelu(x)
        return x


class MultiHeadMamba(nn.Module):
    def __init__(
        self,
        *,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_heads: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
    ) -> None:
        super().__init__()
        _require_model_dependencies()
        if len(hidden_sizes) != num_heads:
            raise ValueError(
                f"Expected exactly {num_heads} Mamba hidden sizes, got {len(hidden_sizes)}: "
                f"{list(hidden_sizes)}"
            )

        self.input_size = input_size
        self.hidden_size = list(hidden_sizes)
        self.output_size = output_size
        self.num_head = num_heads

        # Attribute names intentionally match the legacy state_dict layout.
        self.in_layer = nn.ModuleList(
            [nn.Linear(input_size, self.hidden_size[index]) for index in range(num_heads)]
        )
        self.out_layer = nn.ModuleList(
            [nn.Linear(self.hidden_size[index], output_size) for index in range(num_heads)]
        )
        self.mamba = nn.ModuleList(
            [
                Mamba(
                    d_model=self.hidden_size[index],
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for index in range(num_heads)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.gen_score = nn.Linear(output_size * num_heads, 1)

    def forward(self, x: torch.Tensor, is_training: bool = True) -> torch.Tensor:
        del is_training  # Kept in the signature for compatibility with the old call sites.
        outputs: list[torch.Tensor] = []
        for index in range(self.num_head):
            branch = self.in_layer[index](x)
            branch = branch + self.mamba[index](branch)
            branch = self.out_layer[index](branch)
            branch = branch.permute(0, 2, 1)[:, :, -1]
            branch = self.dropout(branch)
            outputs.append(branch)
        return self.gen_score(torch.cat(outputs, dim=1))


class FinMamba(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        stock_num: int,
        hidden_channels: int,
        out_channels: int,
        gat_layers: int,
        gat_heads: int,
        mamba_hidden_sizes: Sequence[int],
        mamba_output_size: int,
        mamba_num_heads: int,
        mamba_d_state: int,
        mamba_d_conv: int,
        mamba_expand: int,
        market_kernel_sizes: Sequence[int],
        market_init_sparsity: float,
        dropout: float,
        seq_len: int,
    ) -> None:
        super().__init__()
        if out_channels != input_dim:
            raise ValueError(
                "For the active FinMamba path, gat-out-channels must equal input_dim: "
                "the GIB loss compares both tensors and the legacy warm window duplicates "
                "raw features. Use --gat-out-channels auto."
            )
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")

        self.MG = MarketGuideInception(
            input_dim=input_dim,
            kernel_sizes=market_kernel_sizes,
            init_sparsity=market_init_sparsity,
        )
        self.AGG = GAT(
            in_channels=input_dim,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=gat_layers,
            num_heads=gat_heads,
        )
        # The source created this correct graph+raw model, then accidentally overwrote it
        # with an input_dim-only Mamba while forward() still concatenated graph+raw inputs.
        self.MAMBA = MultiHeadMamba(
            input_size=input_dim + out_channels,
            hidden_sizes=mamba_hidden_sizes,
            output_size=mamba_output_size,
            num_heads=mamba_num_heads,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=dropout,
        )

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.out_channels = out_channels
        self.stock_num = stock_num

    def forward(
        self,
        x: torch.Tensor,
        relation: torch.Tensor,
        market_index: torch.Tensor,
        window: list[torch.Tensor],
        is_training: bool = True,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        rate = self.MG(market_index)
        top_k = int(rate.item() * self.stock_num)
        sparse_relation = topk_matrix(relation, top_k)
        assert dense_to_sparse is not None
        edge_index, _ = dense_to_sparse(sparse_relation)

        graph_features = self.AGG(x, edge_index)
        loss = gib_loss(x, graph_features)
        temporal_input = torch.cat((graph_features, x), dim=1)
        window.append(temporal_input)

        if len(window) != self.seq_len:
            raise RuntimeError(
                f"Temporal window has length {len(window)}, expected {self.seq_len}."
            )
        sequence = torch.stack(window).permute(1, 0, 2)
        score = self.MAMBA(sequence, is_training=is_training).squeeze(1)
        window.pop(0)
        return score, window, loss
