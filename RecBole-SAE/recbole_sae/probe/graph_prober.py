"""
Graph-based recommender prober: LightGCN.
User representation = GCN-aggregated embedding (model.forward() output).
"""

from __future__ import annotations
from typing import Tuple

import numpy as np
import torch

from .base_prober import BaseProber


class GraphProber(BaseProber):

    @torch.no_grad()
    def probe(self, model, dataset, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        n_users     = dataset.num(dataset.uid_field)
        user_emb, _ = model.forward()
        return np.arange(n_users), user_emb.cpu().float().numpy()[:n_users]

    @torch.no_grad()
    def get_item_embeddings(self, model, dataset) -> np.ndarray:
        n_items      = dataset.num(dataset.iid_field)
        _, item_emb  = model.forward()
        return item_emb.cpu().float().numpy()[:n_items]
