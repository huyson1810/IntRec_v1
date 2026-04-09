"""
Evaluation metrics (RecSAE §4.4, §5.1):
  - Intra/inter-latent Silhouette cosine similarity
  - Concept coverage at multiple confidence thresholds
  - NDCG@K reconstruction quality
"""

from __future__ import annotations
import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CONFIDENCE_THRESH = 0.8


# ── cosine helpers ──────────────────────────────────────────────────────────

def _unit(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(min=eps)


def compute_latent_similarities(
    latent_acts:     np.ndarray,   # [N, latent_dim]
    item_embeddings: np.ndarray,   # [n_items, dim]
    predicted_items: np.ndarray,   # [N] top-1 item per user
    conf_scores:     Dict[int, float],
    threshold:       float = CONFIDENCE_THRESH,
) -> Tuple[float, float]:
    """
    Intra: mean cosine-sim of top-2 activated items per latent (higher = better).
    Inter: mean pairwise cosine-sim of top-1 items across latents  (lower = better).
    Only latents with confidence >= threshold are included.
    """
    valid = [l for l, s in conf_scores.items() if s >= threshold]
    if len(valid) < 2:
        return float("nan"), float("nan")

    top1_embs: Dict[int, np.ndarray] = {}
    intra_vals: List[float] = []

    for l in valid:
        col     = latent_acts[:, l]
        top_idx = np.argsort(col)[::-1]
        pos     = top_idx[col[top_idx] > 0]
        if len(pos) < 2:
            continue
        e1 = _unit(item_embeddings[predicted_items[pos[0]]])
        e2 = _unit(item_embeddings[predicted_items[pos[1]]])
        intra_vals.append(float(e1 @ e2))
        top1_embs[l] = item_embeddings[predicted_items[pos[0]]]

    if len(top1_embs) < 2:
        return float(np.mean(intra_vals)) if intra_vals else float("nan"), float("nan")

    keys = sorted(top1_embs)
    M    = _unit(np.stack([top1_embs[l] for l in keys]))
    sim  = M @ M.T
    mask = ~np.eye(len(keys), dtype=bool)
    return float(np.mean(intra_vals)), float(sim[mask].mean())


# ── concept coverage ─────────────────────────────────────────────────────────

def concept_coverage(
    conf_scores: Dict[int, float],
    thresholds:  Tuple[float, ...] = (0.6, 0.7, 0.8, 0.9, 1.0),
) -> Dict:
    out = {"total": len(conf_scores)}
    for t in thresholds:
        out[f"P>={t:.1f}"] = sum(1 for s in conf_scores.values() if s >= t)
    return out


# ── NDCG@K ───────────────────────────────────────────────────────────────────

def _dcg(rel: np.ndarray, k: int) -> float:
    r = rel[:k]
    return float((r / np.log2(np.arange(2, len(r) + 2))).sum()) if len(r) else 0.0


def ndcg_at_k(
    user_reps:    np.ndarray,           # [N, dim]
    item_embs:    np.ndarray,           # [n_items, dim]
    ground_truth: Dict[int, List[int]], # row_idx → [relevant item ids]
    k:            int = 10,
    batch:        int = 256,
) -> float:
    vals = []
    for s in range(0, len(user_reps), batch):
        scores = user_reps[s:s+batch] @ item_embs.T
        for bi, ui in enumerate(range(s, min(s+batch, len(user_reps)))):
            rel_set = ground_truth.get(ui, [])
            if not rel_set:
                continue
            order = np.argsort(scores[bi])[::-1]
            rel   = np.array([1 if iid in rel_set else 0 for iid in order])
            ideal = _dcg(np.ones(min(len(rel_set), k)), k)
            if ideal > 0:
                vals.append(_dcg(rel, k) / ideal)
    return float(np.mean(vals)) if vals else 0.0


# ── full report ───────────────────────────────────────────────────────────────

def full_report(
    latent_acts:     np.ndarray,
    user_reps:       np.ndarray,
    recon_reps:      np.ndarray,
    item_embeddings: np.ndarray,
    predicted_items: np.ndarray,
    conf_scores:     Dict[int, float],
    ground_truth:    Dict[int, List[int]],
    ks:              Tuple[int, ...] = (10, 20),
) -> Dict:
    intra, inter = compute_latent_similarities(
        latent_acts, item_embeddings, predicted_items, conf_scores
    )
    coverage = concept_coverage(conf_scores)

    n_dead    = int((latent_acts > 0).any(axis=0).__invert__().sum())
    n_latents = latent_acts.shape[1]

    ndcg_base  = {f"NDCG@{k}": ndcg_at_k(user_reps,  item_embeddings, ground_truth, k) for k in ks}
    ndcg_recon = {f"NDCG@{k}": ndcg_at_k(recon_reps, item_embeddings, ground_truth, k) for k in ks}

    report = {
        "silhouette": {"intra": intra, "inter": inter},
        "coverage":   coverage,
        "sparsity": {
            "latent_dim":    n_latents,
            "dead_latents":  n_dead,
            "pct_dead":      round(n_dead / n_latents * 100, 2),
            "mean_active":   round(float((latent_acts > 0).sum(axis=1).mean()), 2),
        },
        "ndcg_base":  ndcg_base,
        "ndcg_recon": ndcg_recon,
    }

    logger.info("── Evaluation Report ──────────────────────")
    logger.info(f"  Intra={intra:.4f}  Inter={inter:.4f}")
    logger.info(f"  Concepts(P≥0.8)={coverage.get('P>=0.8', 0)}  Total={coverage['total']}")
    for k in ks:
        logger.info(
            f"  NDCG@{k}  base={ndcg_base[f'NDCG@{k}']:.4f}  "
            f"recon={ndcg_recon[f'NDCG@{k}']:.4f}"
        )
    logger.info("───────────────────────────────────────────")
    return report
