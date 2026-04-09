"""
Sequential recommender prober: SASRec (and compatible models like TiMiRec).
User representation = self-attention output on the training interaction sequence.
We pass all training batches and keep the last occurrence per user.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import torch

from .base_prober import BaseProber


class SequentialProber(BaseProber):

    @torch.no_grad()
    def probe(self, model, dataset, train_data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        user_reps: Dict[int, np.ndarray] = {}

        for batch in train_data:
            inter        = batch[0] if isinstance(batch, (list, tuple)) else batch
            item_seq     = inter[model.ITEM_SEQ].to(model.device)
            item_seq_len = inter[model.ITEM_SEQ_LEN].to(model.device)
            user_ids     = inter[model.USER_ID]
            seq_out      = model.forward(item_seq, item_seq_len)   # [B, dim]
            for uid, rep in zip(user_ids.cpu().numpy(), seq_out.cpu().float().numpy()):
                user_reps[int(uid)] = rep

        ids  = np.array(sorted(user_reps.keys()))
        reps = np.array([user_reps[i] for i in ids])
        return ids, reps

    @torch.no_grad()
    def get_item_embeddings(self, model, dataset) -> np.ndarray:
        n_items = dataset.num(dataset.iid_field)
        return model.item_embedding.weight.cpu().float().numpy()[:n_items]
