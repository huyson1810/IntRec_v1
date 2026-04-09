"""
Abstract base prober.  Each supported RecBole model family implements a subclass
that extracts user-interest representations immediately before the prediction layer
(RecSAE paper §3.2.1, eq. 1).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseProber(ABC):
    """
    Subclasses must implement `probe()` which takes a loaded RecBole model +
    data objects and returns (user_ids, representations).
    """

    @abstractmethod
    def probe(self, model, dataset, train_data, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            user_ids        : int ndarray [N]
            representations : float32 ndarray [N, dim]
        """

    def get_item_embeddings(self, model, dataset) -> np.ndarray:
        """Return item embedding matrix [n_items, dim] from the base model."""
        raise NotImplementedError

    def get_item_titles(self, dataset) -> Dict[int, str]:
        """Return {internal_item_id: title_string} from the .item atomic file."""
        import os, re

        id2token: list = dataset.field2id_token.get(dataset.iid_field, [])
        data_path    = dataset.config["data_path"]
        dataset_name = dataset.config["dataset"]
        item_file    = os.path.join(data_path, dataset_name, f"{dataset_name}.item")

        raw: Dict[str, str] = {}
        if os.path.exists(item_file):
            with open(item_file, "r", encoding="utf-8") as fh:
                header     = fh.readline().strip().split("\t")
                item_col   = next((i for i, h in enumerate(header) if h.startswith("item_id")), 0)
                title_col  = next((i for i, h in enumerate(header) if h.startswith("title")), None)
                if title_col is not None:
                    for line in fh:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) > title_col:
                            raw[parts[item_col]] = parts[title_col]

        return {int(iid): raw.get(str(tok), "unknown") for iid, tok in enumerate(id2token)}

    def get_user_history(self, dataset) -> Dict[int, list]:
        """Return {internal_user_id: [internal_item_ids]} from inter_feat."""
        inter    = dataset.inter_feat
        uid_f    = dataset.uid_field
        iid_f    = dataset.iid_field
        history: Dict[int, list] = {}
        for uid, iid in zip(inter[uid_f].numpy(), inter[iid_f].numpy()):
            history.setdefault(int(uid), []).append(int(iid))
        return history
