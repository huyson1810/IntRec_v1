"""
run_neuron_interpret.py
=======================
Interpret the meaning of each neuron/dimension in a trained SAE hidden layer
by finding the top-K most-activated ITEMS per neuron and asking an LLM to
identify the common characteristic that connects them.

Pipeline
--------
  1. Load trained RecBole base-model checkpoint  →  extract item embeddings
  2. Load (or train) RecBole-SAE checkpoint       →  encode item embeddings
                                                       through SAE encoder
  3. Per neuron: rank items by activation value,
     collect top_k item side-information
  4. LLM call  →  2-4 word label summarising the
     dominant characteristic linking those items
  5. Save full per-neuron JSON:
       { neuron_id: { label, top_items: [{rank, item_id, title,
                                          <all side-info fields>,
                                          activation}] } }

Key difference from run_recsae.py
----------------------------------
  run_recsae.py    →  encodes USER representations, interprets via
                       user-interaction sequences (§3.3 of the paper)
  THIS script      →  encodes ITEM embeddings, interprets directly
                       from item side-information (title / genre / year…)

Usage
-----
  # Option A: train SAE fresh + interpret
  python run_neuron_interpret.py \\
      --checkpoint saved/BPR-ml-100k-xxx.pth \\
      --sae_config  configs/sae_default.yaml \\
      --item_category movies

  # Option B: load already-trained SAE + interpret only
  python run_neuron_interpret.py \\
      --checkpoint  saved/BPR-ml-100k-xxx.pth \\
      --sae_checkpoint saved/BPR-ml-100k-xxx-SAE.pth \\
      --item_category movies

  # Option C: OpenAI backend
  python run_neuron_interpret.py \\
      --checkpoint saved/SASRec-ml-100k-xxx.pth \\
      --llm_backend openai --llm_model gpt-4o-mini \\
      --openai_api_key sk-...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SAE neuron interpretation via item side-information + LLM"
    )
    # --- required ---
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to trained RecBole base-model .pth"
    )

    # --- data path override (for different environments) ---
    p.add_argument(
        "--data_path", default=None,
        help="Override dataset path (useful for Kaggle/Colab/different environments). "
             "If not provided, uses path from checkpoint config."
    )

    # --- SAE source (mutually exclusive: train OR load) ---
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--sae_checkpoint",
        help="Path to an already-trained RecBole-SAE .pth  (skips SAE training)"
    )
    g.add_argument(
        "--sae_config", default="configs/sae_default.yaml",
        help="SAE config YAML  (used when training SAE from scratch)"
    )
    # --- output ---
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--output_file", default=None,
                   help="Override output JSON filename (default: <run>-neurons.json)")

    # --- interpretation ---
    p.add_argument("--top_k",           type=int, default=10,
                   help="Top-K items to show per neuron for LLM labelling (default 10)")
    p.add_argument("--min_activation",  type=float, default=0.0,
                   help="Skip neurons where max item activation < this threshold")
    p.add_argument("--neuron_limit",    type=int, default=None,
                   help="Only interpret the first N neurons (useful for quick tests)")

    # --- LLM ---
    p.add_argument("--llm_backend",    choices=["huggingface", "openai"],
                   default="huggingface")
    p.add_argument("--llm_model",
                   default="Qwen/Qwen2.5-1.5B-Instruct",
                   help="HF model ID or OpenAI model name. "
                        "Default Qwen2.5-1.5B-Instruct: 1.5B params, fits in 3GB "
                        "RAM, fast on CPU, excellent for short labelling tasks. "
                        "Use meta-llama/Meta-Llama-3-8B-Instruct if you have a GPU.")
    p.add_argument("--hf_batch_size",  type=int, default=32,
                   help="Number of prompts per HF pipeline forward pass (default 32). "
                        "Increase on high-VRAM GPUs; decrease if OOM.")
    
    # Avoid sending all neurons in one giant list (prevents peak memory spikes)
    p.add_argument("--llm_chunk_size", type=int, default=64,
                   help="How many neuron prompts to send per label_batch() call. "
                        "Lower this if you see CUDA OOM (e.g. 16/32/64).")

    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Max tokens generated per neuron label. "
                        "Lower reduces VRAM/latency (recommended 8–32).")


    p.add_argument("--openai_api_key", default=None)
    p.add_argument("--openai_base_url",default=None)
    p.add_argument("--llm_sleep",      type=float, default=0.0,
                   help="Seconds to sleep between OpenAI API calls (rate limiting). "
                        "Not needed for HuggingFace backend.")

    # --- item context ---
    p.add_argument("--item_category",  default="items",
                   help="Human-readable category name used in prompts, e.g. 'movies'")

    # --- SAE training overrides (ignored when --sae_checkpoint is given) ---
    p.add_argument("--sae_scale",      type=int)
    p.add_argument("--sae_k",          type=int)
    p.add_argument("--sae_epochs",     type=int)
    p.add_argument("--sae_lr",         type=float)
    p.add_argument("--sae_batch_size", type=int)
    p.add_argument("--sae_alpha",      type=float)
    p.add_argument("--sae_dead_window",type=int)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Item side-information loader
# ═══════════════════════════════════════════════════════════════════════════

def load_item_side_info(dataset) -> Dict[int, Dict[str, str]]:
    """
    Read ALL columns from the .item atomic file and return:
        { internal_item_id: { field_name: value, ... } }

    The title field (first text column) is always included.
    Genre, year, director, cast, etc. are included if present.
    Padding / unknown token (index 0) is excluded.
    """
    id2token: list = dataset.field2id_token.get(dataset.iid_field, [])
    token2id = {str(tok): int(iid) for iid, tok in enumerate(id2token)}

    data_path    = dataset.config["data_path"]
    dataset_name = dataset.config["dataset"]
    # item_file    = os.path.join(data_path, dataset_name, f"{dataset_name}.item")
    item_file    = os.path.join(data_path, f"{dataset_name}.item")

    side_info: Dict[str, Dict[str, str]] = {}   # token_str → {field: value}

    if os.path.exists(item_file):
        with open(item_file, "r", encoding="utf-8") as fh:
            raw_header = fh.readline().strip().split("\t")
            # Strip RecBole type suffixes  e.g.  "genres:token_seq" → "genres"
            header = [h.split(":")[0] for h in raw_header]
            item_col = next((i for i, h in enumerate(header)
                             if h.lower() in ("item_id", "itemid", "movie_id", "asin")), 0)

            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(header):
                    parts += [""] * (len(header) - len(parts))
                token = parts[item_col]
                rec = {}
                for i, field in enumerate(header):
                    if i != item_col and parts[i].strip():
                        rec[field] = parts[i].strip()
                if rec:
                    side_info[token] = rec
    else:
        logger.warning(f"Item file not found: {item_file}  (only item_id will be available)")

    # Map to internal integer ids
    result: Dict[int, Dict[str, str]] = {}
    for iid, tok in enumerate(id2token):
        if iid == 0:          # padding token
            continue
        result[iid] = side_info.get(str(tok), {"title": f"item_{tok}"})

    return result


#1. The Dark Knight (2008)  [Action, Crime, Drama]  act=0.847 
def format_item_for_prompt(
    item_info: Dict[str, str],
    rank: int,
    activation: float,
    max_fields: int = 10,
) -> str:
    """
    Format one item's metadata into a compact prompt line.

    Output example:
        1. "title": "Organic Extra Virgin Olive"  |  "category": "Grocery & Gourmet Food"  |  "brand": "Kirkland Signature"  |  "details": "Organic Extra Virgin Olive Oil, 2 L"  |  activation=0.847
    """
    # Priority order for side-info fields to include
    # priority = ["title", "genres", "genre", "year", "release_year",
                # "director", "cast", "tags", "categories", "brand"]

    # amazon grocery and gourmet food
    # priority = ["title", "category", "brand", "details", "description", "price", "popularity_rank"]  
    # priority = ["title", "category", "brand", "description", "price", "popularity_rank"] 
    priority = ["title", "category", "brand", "price", "popularity_rank"]
    
    parts: List[str] = []
    used = 0
    for field in priority:
        if field in item_info and used < max_fields:
            value = item_info[field]
            # Format as "field": "value"
            parts.append(f'"{field}": "{value}"')
            used += 1

    # Fallback: include any remaining fields up to max_fields

    # for field, val in item_info.items():
    #     if field not in priority and used < max_fields:
    #         parts.append(f"{field}={val}")
    #         used += 1

    description = "  |  ".join(parts) if parts else "unknown"
    return f"  {int(rank):2d}. {description}  [activation={activation:.4f}]"


# ═══════════════════════════════════════════════════════════════════════════
# SAE checkpoint loader
# ═══════════════════════════════════════════════════════════════════════════

def load_sae_from_checkpoint(path: str, device: str = "cpu"):
    """Load a RecSAE model from a saved .pth file produced by SAETrainer."""
    from recbole_sae.model import RecSAE

    ckpt      = torch.load(path, map_location=device)
    input_dim = ckpt["input_dim"]
    latent_dim = ckpt["latent_dim"]
    k          = ckpt.get("k", 8)
    scale      = latent_dim // input_dim

    model = RecSAE(input_dim=input_dim, scale=scale, k=k).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()
    logger.info(
        f"Loaded SAE from {path}  "
        f"(input={input_dim}, latent={latent_dim}, k={k})"
    )
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Item activation computation
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def encode_items(
    sae_model,
    item_embs: np.ndarray,
    device:    str = "cpu",
    batch:     int = 512,
) -> np.ndarray:
    """
    Pass item embeddings through the SAE encoder.

    Returns
    -------
    item_acts : float32 ndarray [n_items, latent_dim]
        Raw (non-discretised) activation values.
        Each row has exactly k non-zero entries.
    """
    sae_model.eval()
    parts = []
    for s in range(0, len(item_embs), batch):
        x   = torch.tensor(item_embs[s:s+batch], dtype=torch.float32).to(device)
        act = sae_model.encode(x)
        parts.append(act.cpu().numpy())
    return np.concatenate(parts, axis=0)   # [n_items, latent_dim]


# ═══════════════════════════════════════════════════════════════════════════
# LLM backend
# ═══════════════════════════════════════════════════════════════════════════

class _LLMBackend:
    def label_batch(self, all_messages: List[List[Dict]], max_new_tokens: int = 32) -> List[str]:
        """Label a batch of neurons in one call. Returns list of raw label strings."""
        raise NotImplementedError


class _HuggingFaceBackend(_LLMBackend):
    """
    HuggingFace text-generation backend with batched inference.

    Bug fix vs original: the previous code called pipeline() once per neuron
    in a Python loop. On CPU, Llama-3-8B runs at ~2 tok/s, so with 300-token
    prompts that is 2-3 minutes per neuron. With swap-to-disk (model > RAM):
    30+ minutes per neuron.

    Fixes:
    1. Collect ALL prompts, then call pipeline() ONCE with batch_size=N.
       The pipeline handles tokenization and batched forward passes internally.
    2. Default model changed to Qwen2.5-1.5B-Instruct — ~12x smaller than
       Llama-3-8B, fits in 3 GB RAM at fp16, still excellent for 2-4 word
       classification labels. Change --llm_model if you have a GPU.
    3. Explicit CPU warning: if no GPU is detected, strongly suggest a small
       quantized model or the OpenAI backend.
    """

    def __init__(self, model_name: str, device: str, hf_batch_size: int = 32):
        import torch
        from transformers import pipeline as hf_pipeline

        if device == "cpu":
            logger.warning(
                "No GPU detected — running HuggingFace LLM on CPU. "
                "Llama-3-8B on CPU can take 30+ min/neuron due to RAM pressure. "
                "Recommendations:\n"
                "  (a) Use a small model:  --llm_model Qwen/Qwen2.5-1.5B-Instruct\n"
                "  (b) Use OpenAI:         --llm_backend openai --llm_model gpt-4o-mini\n"
                "  (c) Use a CUDA GPU."
            )

        logger.info(f"Loading HF model: {model_name}  (batch_size={hf_batch_size})")
        self._pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            # device_map="auto" handles multi-GPU or CPU+GPU offload automatically
            device_map="auto",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        # Pad to the left so all sequences in a batch end at the same position
        # (important for decoder-only models like Llama / Qwen)
        if self._pipe.tokenizer.pad_token_id is None:
            self._pipe.tokenizer.pad_token_id = self._pipe.tokenizer.eos_token_id
        self._pipe.tokenizer.padding_side = "left"
        self._hf_batch_size = hf_batch_size

    def label_batch(self, all_messages: List[List[Dict]], max_new_tokens: int = 32) -> List[str]:
        """
        Run all prompts in one batched pipeline call.
        The pipeline tokenizes, pads, and processes them together — a single
        forward pass per micro-batch of size self._hf_batch_size.
        """
        tokenizer = self._pipe.tokenizer
        
        # Convert chat messages -> a single prompt string using the model's chat template (if present)    
        prompts: List[str] = []
        for msgs in all_messages:
            if hasattr(tokenizer, "apply_chat_template"):
                prompts.append(
                    tokenizer.apply_chat_template(
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            else:
                # fallback: simple concat
                prompts.append("\n".join(f"{m['role']}: {m['content']}" for m in msgs))

        # IMPORTANT: inference_mode reduces memory vs no_grad for generation workloads
        with torch.inference_mode():
            outputs = self._pipe(
                prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_full_text=False,
                batch_size=self._hf_batch_size,
                pad_token_id=tokenizer.pad_token_id,
            )

        # outputs: List[ [{"generated_text": "..."}] ]  (one list per prompt)
        texts = [out[0]["generated_text"].strip() for out in outputs]

        # Aggressive cleanup to avoid fragmentation over many iterations
        del outputs
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

        return texts


class _OpenAIBackend(_LLMBackend):
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str]):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model  = model_name

    def label_batch(self, all_messages: List[List[Dict]], max_new_tokens: int = 32) -> List[str]:
        """Call OpenAI API sequentially (no native batch endpoint in standard API)."""
        labels = []
        for messages in all_messages:
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature=0.0,
                )
                labels.append(resp.choices[0].message.content.strip())
            except Exception as exc:
                logger.warning(f"OpenAI call failed: {exc}")
                labels.append("unlabelled")
        return labels


def _make_llm(args) -> _LLMBackend:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.llm_backend == "openai":
        if not args.openai_api_key:
            raise ValueError("--openai_api_key is required when --llm_backend openai")
        return _OpenAIBackend(args.llm_model, args.openai_api_key, args.openai_base_url)
    hf_batch = getattr(args, "hf_batch_size", 32)
    return _HuggingFaceBackend(args.llm_model, device, hf_batch_size=hf_batch)


# ═══════════════════════════════════════════════════════════════════════════
# Neuron labelling prompt
# ═══════════════════════════════════════════════════════════════════════════

def _build_labelling_prompt(
    neuron_id:     int,
    item_lines:    List[str],
    item_category: str,
) -> List[Dict]:
    """
    Construct the messages list for a single neuron labelling call.

    System message sets the task; user message lists the top-K items.
    The model must reply with ONLY a 3-5 word label with its confidence scores based on the highly activated items , no explanation.
    """
    system = (
        f"You are analysing neurons in a recommendation model that recommends {item_category}. "
        f"Each neuron activates strongly for {item_category} that share a specific common concept. "
        f"Your goal is to identify the keyword set of the most prominent concepts that capture these {item_category} items (especially the highly activated items at the top of the list). "
        f"Respond only 3-5 concepts with your confidence scores (in the range from 0 to 10) that exactly cover the features of items (brand, category, popular/unpopular, name, expensive/ cheap price, ingredients, etc.)."   
        f"Basically, consider this task as a classification task. Use consistent terms (avoid synonyms for same concept - e.g do NOT use 'sugary', 'sweet' and 'sweetness' for the same concept) !"   
        f"Example: 'Luxurious (confidence: 8), High Quality (confidence: 6), Popular (confidence: 0.4)' or 'Cheap price (confidence: 9), Fish (confidence: 7), High Protein (confidence: 5)'."
        f"Do NOT include <think> or any reasoning. Output only the final labels."                                                                                                                                                                                                           
    )

    items_block = "\n".join(item_lines)
    user = (
        f"Neuron {neuron_id} — top activated {item_category}:\n"
        f"{items_block}\n\n"
        f"Label (3-5 word labels with its confidence scores based on the highly activated items):"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def _clean_label_0(raw: str) -> str:
    """Strip punctuation artefacts and keep only the first line."""
    label = raw.splitlines()[0].strip()
    # Remove any leading/trailing quotes the model might add
    label = label.strip("\"'`")
    return label if label else "unlabelled"

def _clean_label(raw: str) -> str:
    """
    Clean the model output into a compact label.

    - Strips common chain-of-thought wrappers like <think>...</think>
    - Keeps the first non-empty line
    - Removes surrounding quotes/backticks
    """
    if raw is None:
        return "unlabelled"

    text = str(raw).strip()

    # Remove <think>...</think> blocks if present
    # (some models emit these even when prompted not to)
    while True:
        start = text.find("<think>")
        end = text.find("</think>")
        if start != -1 and end != -1 and end > start:
            text = (text[:start] + text[end + len("</think>"):]).strip()
        else:
            break

    # If the model outputs "<think>" without closing tag, drop that line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln not in ("<think>", "</think>")]

    if not lines:
        return "unlabelled"

    # label = lines[0].strip().strip("\"'`")
    # return label if label else "unlabelled"

    return lines[0].strip().strip("\"'`")

# Evaluate the SAE model
def evaluate_sae(
    sae_model,
    test_embs: np.ndarray,
    device: str = "cpu",
    batch: int = 512,
) -> Dict[str, float]:
    """
    Evaluate SAE reconstruction quality and sparsity on test embeddings.
    
    Returns
    -------
    metrics : dict with keys:
        - mse_loss: mean squared error of reconstruction
        - l0_norm: average number of non-zero activations per sample
        - cosine_sim: average cosine similarity between input and reconstruction
    """
    sae_model.eval()
    
    mse_losses = []
    l0_norms = []
    cosine_sims = []
    
    with torch.no_grad():
        for s in range(0, len(test_embs), batch):
            x = torch.tensor(test_embs[s:s+batch], dtype=torch.float32).to(device)
            
            # Encode and reconstruct
            acts = sae_model.encode(x)
            x_recon = sae_model.decode(acts)
            
            # MSE loss
            mse = torch.mean((x - x_recon) ** 2)
            mse_losses.append(mse.item())
            
            # L0 norm (sparsity)
            l0 = torch.mean((acts > 0).float().sum(dim=1))
            l0_norms.append(l0.item())
            
            # Cosine similarity
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            recon_norm = torch.nn.functional.normalize(x_recon, p=2, dim=1)
            cos_sim = torch.mean(torch.sum(x_norm * recon_norm, dim=1))
            cosine_sims.append(cos_sim.item())
    
    return {
        "mse_loss": float(np.mean(mse_losses)),
        "l0_norm": float(np.mean(l0_norms)),
        "cosine_similarity": float(np.mean(cosine_sims)),
    }


def analyze_neuron_coverage(item_acts: np.ndarray, latent_dim: int) -> Dict:
    """
    Analyze how neurons are utilized across the item embedding space.
    
    Returns statistics about neuron activation patterns.
    """
    # Neurons with at least one active item
    active_neurons = np.sum((item_acts > 0).any(axis=0))
    
    # Mean activation per neuron (for active items)
    mean_acts_per_neuron = np.array([
        np.mean(item_acts[item_acts[:, n] > 0, n]) if np.any(item_acts[:, n] > 0) else 0
        for n in range(latent_dim)
    ])
    
    # Max activation per neuron
    max_acts_per_neuron = np.max(item_acts, axis=0)
    
    # Frequency of activation (% of items that activate each neuron)
    freq_per_neuron = np.mean((item_acts > 0), axis=0) * 100
    
    return {
        "active_neurons": int(active_neurons),
        "dead_neurons": int(latent_dim - active_neurons),
        "neuron_utilization": float(active_neurons / latent_dim * 100),
        "mean_max_activation": float(np.mean(max_acts_per_neuron)),
        "mean_activation_freq": float(np.mean(freq_per_neuron)),
        "max_activation_freq": float(np.max(freq_per_neuron)),
        "min_activation_freq": float(np.min(freq_per_neuron[freq_per_neuron > 0])) if np.any(freq_per_neuron > 0) else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Run: {run_name}   category: {args.item_category}")

    # ── Step 1: Load RecBole model + extract item embeddings ──────────────
    logger.info("── Step 1: Loading base model ──────────────────")
    from recbole_sae.probe import probe_model

    # Override data_path if provided
    if args.data_path:
        logger.info(f"Using custom data_path: {args.data_path}")
        # Pass data_path to probe_model (check if it accepts this parameter)
        data = probe_model(args.checkpoint, data_path=args.data_path)
    else:
        data = probe_model(args.checkpoint)

    # data        = probe_model(args.checkpoint)
    model_name  = data["model_name"]
    item_embs   = data["item_embeddings"]   # [n_items, dim]
    dataset     = data["dataset"]
    input_dim   = item_embs.shape[1]
    user_embs   = data["representations"]   # [N users, dim]

    # dataset_name = "amazon-grocery"
    dataset_name = dataset.config["dataset"]
    #data_path    = dataset.config["data_path"]
    logger.info(
        f"Model={model_name} uses={len(user_embs)} items={len(item_embs)}  dim={input_dim}"
    )

    item_file = '/content/drive/MyDrive/Colab_Notebooks/CIKMv3/RecBole/dataset/amazon-grocery/amazon-grocery.item'
    # Load rich item side-information (all .item columns)
    item_side_info = load_item_side_info(dataset)   # {item_id: {field: value}}
    logger.info(
        f"Item side-info loaded for {len(item_side_info)} items  "
        f"(fields: {list(next(iter(item_side_info.values()), {}).keys())})"
    )

    # Specify the output file path of item_side_info
 
    # output_item_side_info= f"./results/{dataset_name}_item_side_info.json"

    # Save the object to a JSON file
 
    # with open(output_item_side_info, 'w') as f:
    #     json.dump(item_side_info, f, indent=4)
    # logger.info(f"Object saved to {output_item_side_info}")

    # ── Step 2: Get/train SAE ─────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.sae_checkpoint:
        logger.info(f"── Step 2: Loading SAE from {args.sae_checkpoint} ──")
        sae_model = load_sae_from_checkpoint(args.sae_checkpoint, device)
    else:
        logger.info("── Step 2: Training SAE ────────────────────────")
        # Load YAML config and apply CLI overrides
        cfg_path = args.sae_config if args.sae_config else "configs/sae_default.yaml"
        with open(cfg_path) as fh:
            sae_cfg = yaml.safe_load(fh)
        for key in ("sae_scale", "sae_k", "sae_epochs", "sae_lr",
                    "sae_batch_size", "sae_alpha", "sae_dead_window"):
            val = getattr(args, key, None)
            if val is not None:
                sae_cfg[key] = val

        from recbole_sae.trainer import SAETrainer

        # Bug fix: original code trained SAE on USER reps then encoded ITEMS.
        # LightGCN user embeddings (multi-hop graph aggregation) and item
        # embeddings (direct lookup) have very different distributions.
        # Encoding items through a user-trained SAE means items land in
        # untrained regions of the encoder feature space → most neurons never
        # fire on any item → n_active_neurons collapses to ~49.
        # Fix: train SAE directly on item_embs so the encoder learns to spread
        # the item distribution across all latent neurons.
        
        # logger.info("Training SAE on ITEM embeddings (not user reps)")
        
        # Train SAE on BOTH user AND item embeddings
        logger.info("Training SAE on BOTH user and item embeddings")
        combined_embs = np.vstack([user_embs, item_embs])
        logger.info(
            f"Combined embeddings shape: {combined_embs.shape}  "
            f"({len(user_embs)} users + {len(item_embs)} items)"
        )
        
        
        trainer = SAETrainer(
            input_dim         = input_dim,
            scale             = sae_cfg.get("sae_scale",          16),
            k                 = sae_cfg.get("sae_k",               8),
            lr                = sae_cfg.get("sae_lr",          5e-5),
            batch_size        = sae_cfg.get("sae_batch_size",     256),
            epochs            = sae_cfg.get("sae_epochs",         200),
            alpha             = sae_cfg.get("sae_alpha",      1/32.0),
            dead_window       = sae_cfg.get("sae_dead_window",    400),
            normalize_inputs  = sae_cfg.get("normalize_inputs",  True),
            device            = device,
            save_dir          = "saved",
        )
        
        # sae_model = trainer.train(item_embs, run_name=run_name)
        sae_model = trainer.train(combined_embs, run_name=run_name)
            
        sae_model.eval()
        # Keep trainer reference so encode_items uses the same normalization
        _trainer_ref = trainer

    latent_dim = sae_model.latent_dim
    logger.info(f"SAE latent_dim={latent_dim}  k={sae_model.k}")

    # ── Step 3: Encode item embeddings through SAE ────────────────────────
    logger.info("── Step 3: Encoding item embeddings through SAE ─")
    item_acts = encode_items(sae_model, item_embs, device=device)
    # item_acts : [n_items, latent_dim]  raw float32 activations

    logger.info(
        f"item_acts shape={item_acts.shape}  "
        f"non-zero={float((item_acts > 0).mean())*100:.1f}%"
    )

    # ── Step 3.5: Evaluate SAE performance ──────────────────────────────
    logger.info("── Step 3.5: Evaluating SAE ───────────────────")
    eval_metrics = evaluate_sae(sae_model, item_embs, device=device)
    neuron_coverage = analyze_neuron_coverage(item_acts, latent_dim)
    
    logger.info(f"Reconstruction MSE: {eval_metrics['mse_loss']:.6f}")
    logger.info(f"Avg sparsity (L0): {eval_metrics['l0_norm']:.2f} / {latent_dim}")
    logger.info(f"Cosine similarity: {eval_metrics['cosine_similarity']:.4f}")
    logger.info(f"Active neurons: {neuron_coverage['active_neurons']}/{latent_dim} "
                f"({neuron_coverage['neuron_utilization']:.1f}%)")
    logger.info(f"Neuron activation frequency: {neuron_coverage['mean_activation_freq']:.2f}% "
                f"(min={neuron_coverage['min_activation_freq']:.2f}%, "
                f"max={neuron_coverage['max_activation_freq']:.2f}%)")
    


    # ── Step 4: Build per-neuron top-K item list ──────────────────────────
    logger.info("── Step 4: Ranking top items per neuron ────────")
    n_neurons = latent_dim
    n_limit   = args.neuron_limit if args.neuron_limit else n_neurons
    top_k     = args.top_k

    # Pre-compute: for each neuron, sorted item indices by descending activation
    # Store as a list of lists for JSON serialisation
    neuron_top_items: Dict[int, List[Dict]] = {}

    for neuron in range(n_limit):
        col         = item_acts[:, neuron]          # [n_items]
        max_act     = float(col.max())

        if max_act < args.min_activation:
            continue   # dead or near-dead neuron — skip

        top_indices = np.argsort(col)[::-1][:top_k]  # descending

        items_for_neuron = []
        for rank, item_id in enumerate(top_indices, start=1):
            act_val  = float(col[item_id])
            if act_val <= 0:
                break   # all remaining are zero (Top-K SAE)

            side     = item_side_info.get(int(item_id), {})
            entry    = {
                "rank":       rank,
                "item_id":    int(item_id),
                "activation": round(act_val, 6),
            }
            entry.update(side)   # merge all side-info fields inline
            items_for_neuron.append(entry)

        if items_for_neuron:
            neuron_top_items[neuron] = items_for_neuron

    active_neurons = len(neuron_top_items)
    logger.info(
        f"Active neurons (≥1 item): {active_neurons}/{n_limit}  "
        f"(dead: {n_limit - active_neurons})"
    )
    
    # ── Save Step-4 intermediate (top items per neuron) BEFORE any LLM calls ──
    top_items_path = os.path.join(args.output_dir, f"{dataset_name}_{run_name}-neuron_top_items.json")
    with open(top_items_path, "w", encoding="utf-8") as fh:
        # json keys must be strings for portability
        json.dump({str(k): v for k, v in neuron_top_items.items()}, fh, indent=2, ensure_ascii=False)
    logger.info(f"Saved neuron_top_items → {top_items_path}")

    # ── Step 5: LLM labelling ─────────────────────────────────────────────
    logger.info("── Step 5: LLM neuron labelling ────────────────")
    llm = _make_llm(args)

    # Build ALL prompts first, then label in one batched call.
    # Bug fix: original code called llm.chat() once per neuron inside a loop.
    # With 1024 active neurons and a CPU-only Llama-3-8B that is O(hours).
    # Fix: collect every prompt → call label_batch() once → pipeline processes
    # them in micro-batches of hf_batch_size, reusing the KV cache prefix.
    ordered_neurons = sorted(neuron_top_items.keys())
    all_prompts: List[List[Dict]] = []
    for neuron in ordered_neurons:
        items = neuron_top_items[neuron]
        item_lines = []
        for entry in items:
            side = {k: v for k, v in entry.items()
                    if k not in ("rank", "item_id", "activation")}
            item_lines.append(
                format_item_for_prompt(side, entry["rank"], entry["activation"])
            )
        all_prompts.append(_build_labelling_prompt(neuron, item_lines, args.item_category))

    # logger.info(f"Sending {len(all_prompts)} prompts to LLM in one batched call …")
    
    # reduce generation length: Change 32 → 8 or 16 to reduce KV-cache growth during decoding.
    # raw_labels = llm.label_batch(all_prompts, max_new_tokens=12)


    # ---- Build output skeleton BEFORE labelling so we can save progress ----
    max_act_per_neuron = {
        neuron: round(float(max(e["activation"] for e in items)), 6)
        for neuron, items in neuron_top_items.items()
    }

    output: Dict = {
        "run_name":     run_name,
        "base_model":   model_name,
        "dataset":      dataset.config["dataset"],
        "item_category":args.item_category,
        "sae_latent_dim": latent_dim,
        "sae_k":          sae_model.k,
        "n_items_encoded": int(item_embs.shape[0]),
        "top_k":           top_k,
        "n_active_neurons": active_neurons,
        "sae_metrics": {
            "mse_loss": round(eval_metrics["mse_loss"], 6),
            "l0_norm": round(eval_metrics["l0_norm"], 2),
            "cosine_similarity": round(eval_metrics["cosine_similarity"], 4),
        },
        "neuron_coverage": {
            "active_neurons": neuron_coverage["active_neurons"],
            "dead_neurons": neuron_coverage["dead_neurons"],
            "neuron_utilization_pct": round(neuron_coverage["neuron_utilization"], 2),
            "mean_max_activation": round(neuron_coverage["mean_max_activation"], 4),
            "mean_activation_freq_pct": round(neuron_coverage["mean_activation_freq"], 2),
            "max_activation_freq_pct": round(neuron_coverage["max_activation_freq"], 2),
            "min_activation_freq_pct": round(neuron_coverage["min_activation_freq"], 2),
        },
        "neurons": {}
    }

    for neuron in ordered_neurons:
        output["neurons"][str(neuron)] = {
            "neuron_id":      neuron,
            "label":          "unlabelled",
            "max_activation": max_act_per_neuron.get(neuron, 0.0),
            "top_items":      neuron_top_items[neuron],
        }

    # Paths
    fname = args.output_file or f"{run_name}-neurons.json"
    out_path = os.path.join(args.output_dir, fname)

    progress_path = out_path.replace(".json", "-progress.json")
    progress_labels_path = out_path.replace(".json", "-progress-labels.json")

    # ---- Incremental labelling: no raw_labels list ----
    logger.info(f"Labelling {len(all_prompts)} neurons incrementally …")
    neuron_labels: Dict[int, str] = {}

    save_every = 5  # write files every 5 neurons (set to 1 for max safety)

    for idx, (neuron, prompt) in enumerate(zip(ordered_neurons, all_prompts), start=1):
        logger.info(f"  neuron {idx}/{len(ordered_neurons)} (id={neuron})")

        raw = llm.label_batch([prompt], max_new_tokens=args.max_new_tokens)[0]
        cleaned = _clean_label(raw)

        neuron_labels[neuron] = cleaned
        output["neurons"][str(neuron)]["label"] = cleaned

        logger.info(f'    label: "{cleaned[:160]}"')

        # Save progress periodically
        if (idx % save_every) == 0 or idx == len(ordered_neurons):
            with open(progress_path, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2, ensure_ascii=False)

            with open(progress_labels_path, "w", encoding="utf-8") as fh:
                json.dump({str(k): v for k, v in neuron_labels.items()},
                          fh, indent=2, ensure_ascii=False)

            logger.info(f"    progress saved → {progress_path}")

        # extra VRAM fragmentation hygiene
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- Step 6: save final outputs (same filenames as before) ----
    logger.info("── Step 6: Saving results ──────────────────────")

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    summary_path = out_path.replace(".json", "-labels.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({str(k): v for k, v in neuron_labels.items()},
                  fh, indent=2, ensure_ascii=False)

    logger.info(f"Full output   → {out_path}")
    logger.info(f"Label summary → {summary_path}")
    logger.info(f"Done. Labelled {len(neuron_labels)}/{active_neurons} active neurons.")


    # Print a short preview to console
    print("\n── Neuron label preview (first 20 active neurons) ──────────────")
    for neuron in sorted(list(neuron_top_items.keys()))[:20]:
        entry  = output["neurons"][str(neuron)]
        label  = entry["label"]
        top1   = entry["top_items"][0] if entry["top_items"] else {}
        title  = top1.get("title", top1.get("item_id", "?"))
        print(f"  neuron {neuron:4d} | {label:<40s} | top item: {title}")
    print("────────────────────────────────────────────────────────────────\n")


# ── Re-export parse_args so tests / notebooks can import it cleanly
parse_args = _parse_args


if __name__ == "__main__":
    main()
