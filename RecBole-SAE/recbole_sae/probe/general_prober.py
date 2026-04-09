"""
General recommender prober: BPR / BPRMF.
User representation = static user-embedding vector (§3.2.1).
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from .base_prober import BaseProber


class GeneralProber(BaseProber):

    @torch.no_grad()
    def probe(self, model, dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        n_users  = dataset.num(dataset.uid_field)
        user_ids = torch.arange(n_users, device=model.device)
        reps     = model.user_embedding(user_ids)
        return user_ids.cpu().numpy(), reps.cpu().float().numpy()

    @torch.no_grad()
    def get_item_embeddings(self, model, dataset) -> np.ndarray:
        n_items = dataset.num(dataset.iid_field)
        return model.item_embedding.weight.cpu().float().numpy()[:n_items]
