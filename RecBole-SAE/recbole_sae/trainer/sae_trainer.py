"""
RecSAE trainer (§3.2.3).  Wraps the SAE training loop with:
  - dead-latent tracking (steps_since_active > dead_window → auxiliary loss)
  - decoder normalisation after every gradient step
  - best-checkpoint saving
  - input L2-normalisation (critical for LightGCN whose item/user norms vary)

Bug fixes vs original:
  - batch_size default raised from 8 → 256.
    With batch=8 and 944 users: 118 steps/epoch, dead_window=400 fires at
    epoch 3.4 before the aux loss has time to work. batch=256 → 4 steps/epoch
    → latents have many more epochs to be activated before dying.
  - normalize_inputs=True: L2-normalises each input vector before encoding.
    LightGCN item embeddings have varying norms (1–5×). Without normalisation,
    high-norm items dominate top-k every step → same ~50 neurons fire always
    → the other 974 die immediately.
"""

from __future__ import annotations
import logging
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from recbole_sae.model import RecSAE

logger = logging.getLogger(__name__)


class SAETrainer:

    def __init__(
        self,
        input_dim:        int,
        scale:            int   = 16,
        k:                int   = 8,
        lr:               float = 5e-5,
        batch_size:       int   = 256,   # was 8 — see module docstring
        epochs:           int   = 200,   # was 50: more epochs needed with larger batches
        alpha:            float = 1.0 / 32.0,
        dead_window:      int   = 400,
        normalize_inputs: bool  = True,  # L2-normalise every input vector
        device:           str   = "cpu",
        save_dir:         str   = "saved",
    ):
        self.epochs           = epochs
        self.batch_size       = batch_size
        self.alpha            = alpha
        self.dead_window      = dead_window
        self.normalize_inputs = normalize_inputs
        self.device           = device
        self.save_dir         = save_dir

        self.model = RecSAE(input_dim, scale, k).to(device)
        self.opt   = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._steps_since_active = torch.zeros(self.model.latent_dim, dtype=torch.long)

    # ── input normalisation ───────────────────────────────────────────────
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """L2-normalise rows if normalize_inputs is True."""
        if not self.normalize_inputs:
            return x
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _norm_np(self, x: np.ndarray) -> np.ndarray:
        if not self.normalize_inputs:
            return x
        norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-8)
        return (x / norms).astype(np.float32)

    # ── dead-latent tracking ──────────────────────────────────────────────
    def _update_dead(self, acts: torch.Tensor) -> torch.Tensor:
        active = (acts.detach().cpu() > 0).any(dim=0)
        self._steps_since_active[active]  = 0
        self._steps_since_active[~active] += 1
        return self._steps_since_active > self.dead_window

    # ── training loop ─────────────────────────────────────────────────────
    def train(self, representations: np.ndarray, run_name: str = "recsae") -> RecSAE:
        os.makedirs(self.save_dir, exist_ok=True)

        # Apply L2 normalisation once before building the dataloader so every
        # epoch sees normalised vectors without repeated recomputation.
        reps_norm = self._norm_np(representations)

        loader = DataLoader(
            TensorDataset(torch.tensor(reps_norm, dtype=torch.float32)),
            batch_size=self.batch_size, shuffle=True,
        )
        best_loss, best_state = float("inf"), None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            ep_loss = ep_recon = ep_aux = 0.0

            for (batch,) in loader:
                batch = batch.to(self.device)
                acts, recon = self.model(batch)
                dead_mask   = self._update_dead(acts).to(self.device)

                total, l_r, l_a = self.model.loss(batch, acts, recon, dead_mask, self.alpha)
                self.opt.zero_grad()
                total.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.model._normalise_decoder()

                ep_loss += total.item(); ep_recon += l_r.item(); ep_aux += l_a.item()

            n        = len(loader)
            avg      = ep_loss / n
            dead_pct = (self._steps_since_active > self.dead_window).float().mean() * 100
            if epoch % 20 == 0 or epoch == 1:
                logger.info(
                    f"[SAE] epoch {epoch:3d}/{self.epochs}  "
                    f"loss={avg:.4f}  recon={ep_recon/n:.4f}  "
                    f"aux={ep_aux/n:.4f}  dead={dead_pct:.1f}%"
                )
            if avg < best_loss:
                best_loss  = avg
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        self.model.load_state_dict(best_state)
        path = os.path.join(self.save_dir, f"{run_name}-SAE.pth")
        torch.save({
            "state":            best_state,
            "latent_dim":       self.model.latent_dim,
            "input_dim":        self.model.input_dim,
            "k":                self.model.k,
            "normalize_inputs": self.normalize_inputs,
        }, path)
        logger.info(f"[SAE] Saved → {path}")
        return self.model

    # ── inference helpers ─────────────────────────────────────────────────
    @torch.no_grad()
    def encode_all(self, representations: np.ndarray) -> np.ndarray:
        self.model.eval()
        reps_norm = self._norm_np(representations)
        loader = DataLoader(
            TensorDataset(torch.tensor(reps_norm, dtype=torch.float32)),
            batch_size=512,
        )
        parts = [self.model.encode(b.to(self.device)).cpu().numpy() for (b,) in loader]
        return np.concatenate(parts)

    @torch.no_grad()
    def reconstruct_all(self, representations: np.ndarray) -> np.ndarray:
        self.model.eval()
        reps_norm = self._norm_np(representations)
        loader = DataLoader(
            TensorDataset(torch.tensor(reps_norm, dtype=torch.float32)),
            batch_size=512,
        )
        parts = []
        for (b,) in loader:
            _, r = self.model(b.to(self.device))
            parts.append(r.cpu().numpy())
        return np.concatenate(parts)
