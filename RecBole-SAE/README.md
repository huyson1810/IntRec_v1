# RecBole-SAE

> Sparse Autoencoder interpretation layer for [RecBole](https://github.com/RUCAIBox/RecBole).
> Implements **RecSAE** (Wang et al., ACM TOIS 2026) — same extension pattern as RecBole-GNN.

RecBole handles **everything** (base models, datasets, dataloaders, evaluation, configs).  
This package adds **only** what RecBole doesn't have:

| Component | File(s) |
|---|---|
| SAE architecture (Top-K, dead-latent aux loss) | `recbole_sae/model/sae.py` |
| Probing functions per model family | `recbole_sae/probe/` |
| SAE training loop | `recbole_sae/trainer/sae_trainer.py` |
| LLM concept construction + verification | `recbole_sae/interpret/llm_interpreter.py` |
| Silhouette / NDCG evaluation | `recbole_sae/interpret/evaluator.py` |
| Pipeline entry point | `run_recsae.py` |

---

## Installation

```bash
git clone https://github.com/RUCAIBox/RecBole.git
git clone <this-repo>
cd RecBole-SAE
pip install -e .                 # installs recbole-sae + recbole as deps

# LLM backend (pick one):
pip install transformers accelerate   # local Llama-3
pip install openai                    # GPT-4o-mini / compatible API
```

---

## Step 1 — Train a base model (pure RecBole, unchanged)

```bash
# from the RecBole root, or from here — same CLI
python run_recbole.py \
    --model BPR \
    --dataset ml-100k \
    --config_files configs/BPR/ml-100k.yaml
# → saves  saved/BPR-ml-100k-<timestamp>.pth
```

Any RecBole model works (BPR, LightGCN, SASRec, NGCF, …).  
For new model families, add a one-file prober in `recbole_sae/probe/` and register it in `recbole_sae/probe/__init__.py`.

---

## Step 2 — Run the SAE pipeline

```bash
python run_recsae.py \
    --checkpoint saved/BPR-ml-100k-xxx.pth \
    --sae_config  configs/sae_default.yaml \
    --item_category "movies" \
    --item_name "movie"
```

With OpenAI backend:

```bash
python run_recsae.py \
    --checkpoint saved/SASRec-ml-100k-xxx.pth \
    --llm_backend openai \
    --llm_model   gpt-4o-mini \
    --openai_api_key sk-...
```

Results are written to `results/<run_name>-results.json`.

---

## Configuration

`configs/sae_default.yaml` contains **all** RecBole-SAE hyperparameters.  
Every key can be overridden on the CLI (`--sae_scale 32`, `--sae_k 16`, …).

| Key | Default | Paper |
|---|---|---|
| `sae_scale` | 16 | §4.5 |
| `sae_k` | 8 | §4.5 |
| `sae_epochs` | 50 | §4.5 |
| `sae_lr` | 5e-5 | §4.5 |
| `sae_alpha` | 1/32 | eq. 3 |
| `sae_dead_window` | 400 | §3.2 |
| `n_cases` | 5 | §3.3 |
| `llm_backend` | huggingface | §3.3.1 |
| `llm_model` | Llama-3-8B-Instruct | §3.3.1 |

---

## Extending to a new base model

1. Create `recbole_sae/probe/my_prober.py`:

```python
from recbole_sae.probe.base_prober import BaseProber
import numpy as np, torch

class MyProber(BaseProber):
    @torch.no_grad()
    def probe(self, model, dataset, **kwargs):
        # extract user representations before the prediction layer
        reps = model.my_user_encoder(...)
        return np.arange(len(reps)), reps.cpu().numpy()

    @torch.no_grad()
    def get_item_embeddings(self, model, dataset):
        return model.item_embedding.weight.cpu().numpy()
```

2. Register in `recbole_sae/probe/__init__.py`:

```python
from .my_prober import MyProber
_PROBERS["MyModel"] = MyProber()
```

That's it — the rest of the pipeline (SAE training, LLM interpretation, evaluation) runs unchanged.

---

## Output format (`results/<run>-results.json`)

```json
{
  "model": "BPR",
  "report": {
    "silhouette": {"intra": 0.72, "inter": 0.18},
    "coverage":   {"total": 1024, "P>=0.8": 347},
    "sparsity":   {"dead_latents": 12, "mean_active": 7.9},
    "ndcg_base":  {"NDCG@10": 0.2332},
    "ndcg_recon": {"NDCG@10": 0.2298}
  },
  "concepts": {
    "42": {"concept": "action movies from the 1990s", "confidence": 0.91},
    ...
  }
}
```

---

## Citation

```bibtex
@article{wang2026recsae,
  title   = {Understanding Internal Representations of Recommendation Models
             with Sparse Autoencoders},
  author  = {Wang et al.},
  journal = {ACM Transactions on Information Systems},
  year    = {2026},
  doi     = {10.1145/3795529}
}
```
