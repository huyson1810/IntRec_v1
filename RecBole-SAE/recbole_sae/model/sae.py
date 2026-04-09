"""
RecSAE – Sparse AutoEncoder with Top-K activation (Wang et al., 2026, §3.2).

Architecture:
    encode: S_Latent = TopK( W_enc(S_internal − b_pre) + b_enc )
    decode: Ŝ_internal = W_dec · S_Latent + b_pre

Loss (eq. 3):
    L_total = L_recon + α · L_aux
    L_recon = ‖S_internal − Ŝ_internal‖²
    L_aux   = ‖e − ê‖²   where ê uses only dead-latent directions
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RecSAE(nn.Module):
    """
    Args:
        input_dim : dimension of user representation (e.g. 64)
        scale     : latent_dim = input_dim × scale (paper default 16 → 1024)
        k         : top-k active latents per forward pass (paper default 8)
    """

    def __init__(self, input_dim: int, scale: int = 16, k: int = 8):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = input_dim * scale
        self.k          = k

        self.b_pre = nn.Parameter(torch.zeros(input_dim))          # shared pre/post bias
        self.W_enc = nn.Parameter(torch.empty(input_dim, self.latent_dim))
        self.b_enc = nn.Parameter(torch.zeros(self.latent_dim))
        self.W_dec = nn.Parameter(torch.empty(self.latent_dim, input_dim))

        nn.init.xavier_uniform_(self.W_enc)
        nn.init.xavier_uniform_(self.W_dec)
        self._normalise_decoder()

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _normalise_decoder(self) -> None:
        """Unit-normalise each decoder column after every gradient step."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, D] → S_Latent [B, latent_dim]  (exactly k non-zero per row)."""
        pre_acts = (x - self.b_pre) @ self.W_enc + self.b_enc
        topk_v, topk_i = torch.topk(pre_acts, self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, topk_i, topk_v.clamp(min=0.0))
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """S_Latent [B, latent_dim] → Ŝ_internal [B, D]."""
        return acts @ self.W_dec + self.b_pre

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acts  = self.encode(x)
        recon = self.decode(acts)
        return acts, recon

    # ------------------------------------------------------------------ #
    def loss(
        self,
        x:         torch.Tensor,
        acts:      torch.Tensor,
        recon:     torch.Tensor,
        dead_mask: Optional[torch.Tensor] = None,
        alpha:     float = 1.0 / 32.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (total, recon_loss, aux_loss).
        dead_mask: bool [latent_dim], True = latent inactive > dead_window steps.
        """
        recon_loss = ((x - recon) ** 2).sum(dim=-1).mean()

        if dead_mask is not None and dead_mask.any():
            e        = (x - recon).detach()
            pre_acts = (x - self.b_pre) @ self.W_enc + self.b_enc
            dead_pre = pre_acts.clone()
            dead_pre[:, ~dead_mask] = float("-inf")

            # Bug fix: original min(k, dead_count) = min(8, 975) meant only
            # 8 dead neurons could be revived per step — needs 122 steps to
            # touch all dead neurons once, far too slow.
            # Fix: allow up to k*8 so aux pressure covers the dead population
            # much faster while keeping per-step cost bounded.
            n_dead = int(dead_mask.sum().item())
            k_aux  = min(self.k * 8, n_dead)
            tv, ti = torch.topk(dead_pre, k_aux, dim=-1)
            d_acts = torch.zeros_like(pre_acts)
            d_acts.scatter_(-1, ti, tv.clamp(min=0.0))
            e_hat    = d_acts @ self.W_dec
            aux_loss = ((e - e_hat) ** 2).sum(dim=-1).mean()
        else:
            aux_loss = recon_loss.new_zeros(())

        return recon_loss + alpha * aux_loss, recon_loss, aux_loss
