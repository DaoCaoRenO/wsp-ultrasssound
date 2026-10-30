import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath
from .common import Adapter


class Router(nn.Module):
    """Prompt-aware gating mechanism for MoE."""

    def __init__(
        self,
        dim: int,
        num_experts: int,
        prompt_dim: int = 15,
        temperature: float = 1.0,
        noise: float = 0.0,
    ) -> None:
        super().__init__()
        self.prompt_dim = prompt_dim
        self.temperature = temperature
        self.noise = noise
        self.linear = nn.Linear(dim + prompt_dim, num_experts)

    def forward(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 4:  # [B, C, H, W]
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        else:  # [B, N, C]
            B, N, C = x.shape

        N = x.shape[1]

        if self.prompt_dim > 0:
            if prompt is None:
                prompt = torch.zeros(B, N, self.prompt_dim, device=x.device, dtype=x.dtype)
            else:
                if prompt.dim() == 2:
                    prompt = prompt[:, None, :].expand(B, N, -1)
                elif prompt.dim() == 3:
                    if prompt.shape[1] == 1:
                        prompt = prompt.expand(B, N, -1)
                    elif prompt.shape[1] != N:
                        prompt = prompt[:, :1, :].expand(B, N, -1)
                prompt = prompt.to(dtype=x.dtype, device=x.device)

            gate_input = torch.cat([x, prompt], dim=-1)
        else:
            gate_input = x

        logits = self.linear(gate_input.reshape(-1, gate_input.shape[-1])) / self.temperature
        logits = logits.view(B, N, -1)
        if self.noise > 0 and self.training:
            logits = logits + self.noise * torch.randn_like(logits)
        return logits


class MoEAdapter(nn.Module):
    """Adapter layer with Mixture of Experts and top-k sparse routing."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 0.25,
        num_experts: int = 4,
        top_k: int = 1,
        temperature: float = 1.0,
        prompt_dim: int = 0,
        noise: float = 0.0,
        skip_connect: bool = True,
        act_layer: nn.Module = nn.GELU,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.skip_connect = skip_connect
        self.experts = nn.ModuleList(
            [Adapter(dim, mlp_ratio=mlp_ratio, act_layer=act_layer, skip_connect=False)
             for _ in range(num_experts)]
        )
        self.router = Router(
            dim,
            num_experts,
            temperature=temperature,
            noise=noise,
            prompt_dim=prompt_dim,
        )
        self.dropout = nn.Dropout(0.1)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.last_routing = None
        self.last_entropy = None

    def forward(self, x: torch.Tensor, prompt: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        logits = self.router(x, prompt)
        scores = F.softmax(logits, dim=-1)
        entropy = -(scores * (scores + 1e-9).log()).sum(dim=-1).mean()
        topk_val, topk_idx = torch.topk(scores, self.top_k, dim=-1)
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)

        # build sparse gate matrix for aux loss and routing stats
        gates = torch.zeros_like(scores)
        gates.scatter_(-1, topk_idx, topk_val)

        x_flat = x.view(-1, C)
        topk_val_flat = topk_val.view(-1, self.top_k)
        topk_idx_flat = topk_idx.view(-1, self.top_k)
        out = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx_flat == expert_id
            if not mask.any():
                continue
            token_indices, k_indices = mask.nonzero(as_tuple=True)
            tokens = x_flat[token_indices]
            expert_out = expert(tokens)
            weights = topk_val_flat[token_indices, k_indices].unsqueeze(-1)
            out.index_add_(0, token_indices, expert_out * weights)

        out = out.view(B, N, C)
        out = self.dropout(out)
        if self.skip_connect:
            out = out + x
        out = self.drop_path(out)
        self.last_routing = gates.mean(dim=(0, 1)).detach()
        self.last_entropy = entropy.detach()
        ideal = torch.full_like(self.last_routing, 1.0 / self.num_experts)
        self.aux_loss = F.mse_loss(self.last_routing, ideal)
        return out

    def get_aux_loss(self) -> torch.Tensor:
        return self.aux_loss
