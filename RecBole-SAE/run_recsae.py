"""
run_recsae.py – RecBole-SAE pipeline entry point.

Mirrors RecBole-GNN's run_recbole_gnn.py style.
Loads a trained RecBole checkpoint, probes user representations,
trains the SAE, interprets latents via LLM, evaluates results.

Usage:
    python run_recsae.py \\
        --checkpoint saved/BPR-ml-100k-xxx.pth \\
        --sae_config  configs/sae_default.yaml

Optional overrides (same style as RecBole CLI):
    --sae_scale 16 --sae_k 8 --llm_backend openai --item_category "movies"
"""

import argparse
import json
import logging
import os
import sys

import yaml
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="RecBole-SAE pipeline")
    parser.add_argument("--checkpoint",  required=True, help="Path to trained RecBole .pth")
    parser.add_argument("--sae_config",  default="configs/sae_default.yaml")
    parser.add_argument("--output_dir",  default="results")

    # allow any sae_default.yaml key to be overridden on the command line
    parser.add_argument("--sae_scale",           type=int)
    parser.add_argument("--sae_k",               type=int)
    parser.add_argument("--sae_epochs",          type=int)
    parser.add_argument("--sae_lr",              type=float)
    parser.add_argument("--sae_batch_size",      type=int)
    parser.add_argument("--sae_alpha",           type=float)
    parser.add_argument("--sae_dead_window",     type=int)
    parser.add_argument("--llm_backend",         type=str, choices=["huggingface", "openai"])
    parser.add_argument("--llm_model",           type=str)
    parser.add_argument("--openai_api_key",      type=str)
    parser.add_argument("--openai_base_url",     type=str)
    parser.add_argument("--n_cases",             type=int)
    parser.add_argument("--item_category",       type=str)
    parser.add_argument("--item_name",           type=str)
    parser.add_argument("--llm_sleep",           type=float)
    parser.add_argument("--confidence_threshold",type=float)
    return parser.parse_args()


def merge_cfg(base: dict, args) -> dict:
    """Override YAML defaults with any CLI args that were explicitly set."""
    overridable = [
        "sae_scale", "sae_k", "sae_epochs", "sae_lr", "sae_batch_size",
        "sae_alpha", "sae_dead_window", "llm_backend", "llm_model",
        "openai_api_key", "openai_base_url", "n_cases", "item_category",
        "item_name", "llm_sleep", "confidence_threshold",
    ]
    for key in overridable:
        val = getattr(args, key, None)
        if val is not None:
            base[key] = val
    return base


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load SAE config
    with open(args.sae_config) as fh:
        sae_cfg = yaml.safe_load(fh)
    sae_cfg = merge_cfg(sae_cfg, args)

    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Run name: {run_name}")

    # 2. Probe base model (RecBole does all the heavy lifting here)
    logger.info("── Step 1: Probing base model ──────────────────")
    from recbole_sae.probe import probe_model
    data = probe_model(args.checkpoint)

    reps         = data["representations"]   # [N, dim]
    item_embs    = data["item_embeddings"]   # [n_items, dim]
    item_titles  = data["item_titles"]
    user_history = data["user_history"]
    user_ids     = data["user_ids"]
    model_name   = data["model_name"]
    input_dim    = reps.shape[1]

    logger.info(f"Model={model_name}  users={len(reps)}  dim={input_dim}  "
                f"items={len(item_embs)}")

    # 3. Train SAE
    logger.info("── Step 2: Training SAE ────────────────────────")
    from recbole_sae.trainer import SAETrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = SAETrainer(
        input_dim   = input_dim,
        scale       = sae_cfg.get("sae_scale",       16),
        k           = sae_cfg.get("sae_k",            8),
        lr          = sae_cfg.get("sae_lr",       5e-5),
        batch_size  = sae_cfg.get("sae_batch_size",    8),
        epochs      = sae_cfg.get("sae_epochs",       50),
        alpha       = sae_cfg.get("sae_alpha",   1/32.0),
        dead_window = sae_cfg.get("sae_dead_window",  400),
        device      = device,
        save_dir    = "saved",
    )
    sae_cfg["device"] = device
    sae = trainer.train(reps, run_name=run_name)

    # 4. Encode all users → latent activations
    logger.info("── Step 3: Encoding user representations ───────")
    latent_acts = trainer.encode_all(reps)         # [N, latent_dim]
    recon_reps  = trainer.reconstruct_all(reps)    # [N, dim]

    from recbole_sae.interpret import compute_activation_levels, predict_top1
    act_levels      = compute_activation_levels(latent_acts)     # [N, latent_dim] int 0-10
    predicted_items = predict_top1(reps, item_embs)              # [N] int

    # 5. LLM interpretation
    logger.info("── Step 4: LLM concept interpretation ─────────")
    from recbole_sae.interpret import LLMInterpreter

    interpreter = LLMInterpreter(
        cfg             = sae_cfg,
        item_titles     = item_titles,
        user_history    = user_history,
        predicted_items = predicted_items,
        act_levels      = act_levels,
    )
    concepts, conf_scores = interpreter.run()

    # 6. Evaluate
    logger.info("── Step 5: Evaluation ──────────────────────────")
    from recbole_sae.interpret import full_report

    # Build ground truth: user row_index → last-interaction item(s)
    ground_truth: dict = {}
    for row_idx, uid in enumerate(user_ids.tolist()):
        hist = user_history.get(int(uid), [])
        if hist:
            ground_truth[row_idx] = [hist[-1]]

    report = full_report(
        latent_acts     = latent_acts,
        user_reps       = reps,
        recon_reps      = recon_reps,
        item_embeddings = item_embs,
        predicted_items = predicted_items,
        conf_scores     = conf_scores,
        ground_truth    = ground_truth,
        ks              = tuple(sae_cfg.get("ndcg_ks", [10, 20])),
    )

    # 7. Save results
    out_path = os.path.join(args.output_dir, f"{run_name}-results.json")
    payload = {
        "run_name":    run_name,
        "model":       model_name,
        "sae_config":  sae_cfg,
        "report":      report,
        "n_concepts":  len(concepts),
        "concepts":    {
            str(l): {"concept": v["concept"], "confidence": conf_scores.get(l, 0.0)}
            for l, v in concepts.items()
        },
    }
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    logger.info(f"Results saved → {out_path}")


if __name__ == "__main__":
    main()
