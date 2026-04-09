"""
Probe factory: maps RecBole model name → correct prober class, then
runs it and returns all data needed by the SAE training pipeline.
"""

from __future__ import annotations
from typing import Dict

from recbole.quick_start import load_data_and_model

from .base_prober      import BaseProber
from .general_prober   import GeneralProber
from .graph_prober     import GraphProber
from .sequential_prober import SequentialProber

# ── registry ──────────────────────────────────────────────────────────────
_PROBERS: Dict[str, BaseProber] = {
    "BPR":      GeneralProber(),
    "BPRMF":    GeneralProber(),
    "LightGCN": GraphProber(),
    "SASRec":   SequentialProber(),
    "TiMiRec":  SequentialProber(),
}


def get_prober(model_name: str) -> BaseProber:
    prober = _PROBERS.get(model_name)
    if prober is None:
        raise ValueError(
            f"No prober for '{model_name}'. "
            f"Registered: {list(_PROBERS.keys())}. "
            f"Add a new prober in recbole_sae/probe/ and register it here."
        )
    return prober


def probe_model(checkpoint_path: str) -> Dict:
    """
    Load a trained RecBole checkpoint and extract everything the SAE needs.

    Returns dict with:
        model_name      : str
        user_ids        : ndarray [N]
        representations : ndarray [N, dim]
        item_embeddings : ndarray [n_items, dim]
        item_titles     : Dict[int, str]
        user_history    : Dict[int, List[int]]
        dataset / model / config / train_data / valid_data / test_data
    """
    config, model, dataset, train_data, valid_data, test_data = \
        load_data_and_model(checkpoint_path)

    model_name = config["model"]
    model.eval()

    prober = get_prober(model_name)
    user_ids, reps = prober.probe(
        model=model, dataset=dataset, train_data=train_data
    )

    return {
        "model_name":      model_name,
        "user_ids":        user_ids,
        "representations": reps.astype("float32"),
        "item_embeddings": prober.get_item_embeddings(model, dataset).astype("float32"),
        "item_titles":     prober.get_item_titles(dataset),
        "user_history":    prober.get_user_history(dataset),
        "dataset":         dataset,
        "model":           model,
        "config":          config,
        "train_data":      train_data,
        "valid_data":      valid_data,
        "test_data":       test_data,
    }
