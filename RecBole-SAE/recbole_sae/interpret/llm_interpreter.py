"""
LLM-based concept construction and verification (RecSAE §3.3).

Two backends are supported, selected via config["llm_backend"]:

  "huggingface"  – local Llama-3-8B-Instruct (or any HF chat model)
                   set config["llm_model"] = "meta-llama/Meta-Llama-3-8B-Instruct"

  "openai"       – OpenAI-compatible API (GPT-4o-mini, GPT-4, etc.)
                   set config["llm_model"]  = "gpt-4o-mini"
                       config["openai_api_key"] = "sk-..."
                       config["openai_base_url"] = "https://api.openai.com/v1"  (optional)

Pipeline per latent l (paper §3.3):
  1. Find top-2N most-activated (user, predicted_item) pairs.
  2. Split: N construct | N verify-positive.
  3. Sample N non-activated items for verify-negative.
  4. Discretise activations 0-10 (level 0 = inactive, 1-10 = equal-freq bins).
  5. LLM construction: one-shot prompt → single-sentence concept description.
  6. LLM verification: predict activation level on pos+neg test items → confidence score.
"""

from __future__ import annotations
import logging
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────
N_BINS            = 10
N_CASES           = 5       # N in the paper (top-2N → N construct + N verify-pos)
CONFIDENCE_THRESH = 0.8
VERIFY_THRESHOLD  = 5       # predicted level > 5 → positive class


# ═══════════════════════════════════════════════════════════════════════════
# Activation helpers
# ═══════════════════════════════════════════════════════════════════════════

def compute_activation_levels(latent_acts: np.ndarray) -> np.ndarray:
    """
    Discretise continuous activations → int levels 0-10.
    Level 0 = inactive; levels 1-10 = equal-frequency quantile bins
    computed from the non-zero training distribution (paper §3.3).
    """
    levels = np.zeros_like(latent_acts, dtype=np.int8)
    for l in range(latent_acts.shape[1]):
        col  = latent_acts[:, l]
        mask = col > 0
        if mask.sum() < 2:
            continue
        nz   = col[mask]
        qs   = np.percentile(nz, np.linspace(0, 100, N_BINS + 1)[1:])
        qs   = np.unique(qs)
        bins = np.clip(np.digitize(nz, qs, right=False) + 1, 1, N_BINS)
        levels[mask, l] = bins.astype(np.int8)
    return levels


def predict_top1(user_reps: np.ndarray, item_embs: np.ndarray) -> np.ndarray:
    """Dot-product top-1 item per user. Returns [N_users] item-id array."""
    batch = 512
    out   = []
    for s in range(0, len(user_reps), batch):
        scores = user_reps[s:s+batch] @ item_embs.T
        out.append(scores.argmax(axis=-1))
    return np.concatenate(out)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builders  (paper §3.3.1 / §3.3.2)
# ═══════════════════════════════════════════════════════════════════════════

def _build_construction_prompt(
    latent_id:    int,
    act_seqs:     List[str],    # formatted "title (level)" per case
    item_category: str,
    item_name:    str,
    example_seq:  str,
    example_expl: str,
) -> List[Dict]:
    """Return an OpenAI-style messages list for concept construction (one-shot)."""
    system = (
        f"We're studying neurons in a recommendation model that is used to recommend "
        f"{item_category}. Each neuron looks for some particular concepts in "
        f"{item_category}. Look at the parts of the {item_name} the neuron activates "
        f"for and summarize in a single sentence what the neuron is looking for. "
        f"Don't list examples of words.\n"
        f"The activation format is {item_name}<tab>activation. Activation values range "
        f"from 0 to 10. A neuron finding what it's looking for is represented by a "
        f"non-zero activation value. The higher the activation value, the stronger the match."
    )
    user_example = (
        f"Neuron 0\nActivations:\n{example_seq}\n"
        f"Explanation of neuron 0 behavior: the main thing this neuron does is find"
    )
    seq_str = "\n".join(act_seqs)
    user_target = (
        f"Neuron {latent_id}\nActivations:\n{seq_str}\n"
        f"Explanation of neuron {latent_id} behavior: the main thing this neuron does is find"
    )
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_example},
        {"role": "assistant", "content": example_expl},
        {"role": "user",      "content": user_target},
    ]


def _build_verification_prompt(
    latent_id:     int,
    concept:       str,
    test_sequence: str,
    test_last_item: str,
    item_category: str,
    item_name:     str,
    example_concept: str,
    example_sequence: str,
    example_last_item: str,
    example_activation: str,
) -> List[Dict]:
    """Return messages list for activation-level prediction (one-shot)."""
    system = (
        f"We're studying neurons in a recommendation model that is used to recommend "
        f"{item_category}. Each neuron looks for some particular concepts in "
        f"{item_category}. Look at an explanation of what the neuron does, and try to "
        f"predict its activations on a particular {item_name}.\n"
        f"The activation format is {item_name}<tab>activation, and activations range "
        f"from 0 to 10. Most activations will be 0.\n"
        f"Now, we're going to predict the activation of a new neuron on a single "
        f"{item_name}, following the same rules as the examples above."
    )
    user_ex = (
        f"Neuron 0\n"
        f"Explanation of neuron 0 behavior: the main thing this neuron does is find "
        f"{example_concept}\n"
        f"Sequence:\n{example_sequence}\n"
        f"Last {item_name} in the sequence:\n{example_last_item}\n"
        f"Last {item_name} activation, considering the {item_name} in the context "
        f"in which it appeared in the sequence:"
    )
    user_test = (
        f"Neuron {latent_id}\n"
        f"Explanation of neuron {latent_id} behavior: the main thing this neuron does is find "
        f"{concept}\n"
        f"Sequence:\n{test_sequence}\n"
        f"Last {item_name} in the sequence:\n{test_last_item}\n"
        f"Last {item_name} activation, considering the {item_name} in the context "
        f"in which it appeared in the sequence:"
    )
    return [
        {"role": "system",    "content": system},
        {"role": "user",      "content": user_ex},
        {"role": "assistant", "content": example_activation},
        {"role": "user",      "content": user_test},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# LLM backend abstraction
# ═══════════════════════════════════════════════════════════════════════════

class _LLMBackend:
    def chat(self, messages: List[Dict], max_new_tokens: int = 64) -> str:
        raise NotImplementedError


class HuggingFaceBackend(_LLMBackend):
    """
    Local HuggingFace causal-LM (e.g. meta-llama/Meta-Llama-3-8B-Instruct).
    Loaded once on first call.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        logger.info(f"Loading HuggingFace model: {model_name}")
        self._pipe = pipeline(
            "text-generation",
            model=model_name,
            device_map="auto" if device != "cpu" else None,
            torch_dtype="auto",
        )

    def chat(self, messages: List[Dict], max_new_tokens: int = 64) -> str:
        out = self._pipe(
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return out[0]["generated_text"].strip()


class OpenAIBackend(_LLMBackend):
    """
    OpenAI-compatible REST API (GPT-4o-mini, local vLLM, etc.).
    Requires `pip install openai`.
    """

    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI
        self._client     = OpenAI(api_key=api_key, base_url=base_url)
        self._model_name = model_name

    def chat(self, messages: List[Dict], max_new_tokens: int = 64) -> str:
        resp = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()


def _make_backend(cfg: Dict) -> _LLMBackend:
    backend = cfg.get("llm_backend", "huggingface").lower()
    model   = cfg.get("llm_model", "meta-llama/Meta-Llama-3-8B-Instruct")
    device  = cfg.get("device", "cpu")

    if backend == "huggingface":
        return HuggingFaceBackend(model, device)
    elif backend == "openai":
        return OpenAIBackend(
            model_name = model,
            api_key    = cfg.get("openai_api_key", ""),
            base_url   = cfg.get("openai_base_url", None),
        )
    else:
        raise ValueError(f"Unknown llm_backend '{backend}'. Use 'huggingface' or 'openai'.")


# ═══════════════════════════════════════════════════════════════════════════
# Case builder helpers
# ═══════════════════════════════════════════════════════════════════════════

def _format_activation_sequence(
    history_titles: List[str],
    predicted_title: str,
    pred_level: int,
    item_name: str = "product",
) -> Tuple[str, str]:
    """
    Build the item-activation sequence string used in both prompts.

    Returns:
        full_sequence_str  – history items (level 0) + predicted item
        last_item_str      – just the last item title + level
    """
    lines = [f"{t}\t0" for t in history_titles]
    lines.append(f"{predicted_title}\t{pred_level}")
    return "\n".join(lines), f"{predicted_title}\t{pred_level}"


# ═══════════════════════════════════════════════════════════════════════════
# Core interpreter class
# ═══════════════════════════════════════════════════════════════════════════

class LLMInterpreter:
    """
    Build and verify LLM-based concept descriptions for every SAE latent.

    Args:
        cfg            : dict with keys llm_backend, llm_model, [openai_*],
                         item_category (e.g. "grocery food"), item_name (e.g. "product"),
                         n_cases (default 5), llm_sleep (throttle seconds, default 0)
        item_titles    : {internal_item_id: title_string}
        user_history   : {internal_user_id: [internal_item_ids]}
        predicted_items: [N_users] top-1 predicted item per user (int)
        act_levels     : [N_users, latent_dim] discretised 0-10 int array
    """

    def __init__(
        self,
        cfg:             Dict,
        item_titles:     Dict[int, str],
        user_history:    Dict[int, List[int]],
        predicted_items: np.ndarray,
        act_levels:      np.ndarray,
    ):
        self.cfg             = cfg
        self.item_titles     = item_titles
        self.user_history    = user_history
        self.predicted_items = predicted_items
        self.act_levels      = act_levels.astype(np.int8)
        self.n_cases         = cfg.get("n_cases", N_CASES)
        self.item_category   = cfg.get("item_category", "items")
        self.item_name       = cfg.get("item_name", "item")
        self.sleep_s         = cfg.get("llm_sleep", 0.0)
        self._llm: Optional[_LLMBackend] = None   # lazy init

    def _get_llm(self) -> _LLMBackend:
        if self._llm is None:
            self._llm = _make_backend(self.cfg)
        return self._llm

    # ── sequence helpers ──────────────────────────────────────────────────

    def _title(self, iid: int) -> str:
        return self.item_titles.get(iid, f"item_{iid}")

    def _seq_for_user(self, uid: int, predicted_iid: int, level: int) -> Tuple[str, str]:
        """Full sequence string and last-item string for a given user."""
        history = self.user_history.get(uid, [])
        hist_titles = [self._title(i) for i in history[-10:]]   # cap context
        return _format_activation_sequence(
            hist_titles, self._title(predicted_iid), level, self.item_name
        )

    # ── select cases per latent ───────────────────────────────────────────

    def _select_cases(self, l: int):
        """
        Returns:
            construct_users  : user indices for concept construction [N]
            verify_pos_users : user indices for positive verification [N]
            verify_neg_users : user indices for negative verification [N]
            (all as plain Python lists of ints)
        """
        rng      = np.random.default_rng(42 + l)
        col      = self.act_levels[:, l]
        pos_idx  = np.where(col > 0)[0]
        neg_idx  = np.where(col == 0)[0]

        if len(pos_idx) < 2:
            return None, None, None

        # Sort descending by activation level, take top 2N
        sorted_pos = pos_idx[np.argsort(col[pos_idx])[::-1]][: 2 * self.n_cases]
        rng.shuffle(sorted_pos)
        construct  = sorted_pos[:self.n_cases].tolist()
        verify_pos = sorted_pos[self.n_cases: 2 * self.n_cases].tolist()

        n_neg      = min(self.n_cases, len(neg_idx))
        verify_neg = rng.choice(neg_idx, size=n_neg, replace=False).tolist()

        return construct, verify_pos, verify_neg

    # ── one-shot example (fixed across all latents) ───────────────────────

    @staticmethod
    def _example_for_dataset(item_category: str) -> Tuple[str, str, str, str]:
        """
        Returns a canned one-shot example:
        (example_seq, example_expl, example_last_item, example_activation).
        These are dataset-agnostic fallbacks; callers can override.
        """
        example_seq = (
            "The Dark Knight (2008)\t0\n"
            "Inception (2010)\t0\n"
            "Interstellar (2014)\t8"
        )
        example_expl  = "science-fiction or mind-bending thriller films directed by Christopher Nolan"
        example_last  = "Interstellar (2014)\t8"
        example_activ = "8"
        return example_seq, example_expl, example_last, example_activ

    # ── construction ─────────────────────────────────────────────────────

    def construct_concept(self, l: int, construct_users: List[int]) -> Optional[str]:
        """
        Call LLM once to produce a single-sentence concept description for latent l.
        Returns None on failure.
        """
        llm = self._get_llm()

        act_seqs = []
        for uid in construct_users:
            pred_iid = int(self.predicted_items[uid])
            level    = int(self.act_levels[uid, l])
            seq_str, _ = self._seq_for_user(uid, pred_iid, level)
            act_seqs.append(seq_str)

        ex_seq, ex_expl, _, _ = self._example_for_dataset(self.item_category)

        messages = _build_construction_prompt(
            latent_id     = l,
            act_seqs      = act_seqs,
            item_category = self.item_category,
            item_name     = self.item_name,
            example_seq   = ex_seq,
            example_expl  = ex_expl,
        )
        try:
            raw = llm.chat(messages, max_new_tokens=80)
            # Strip any continuation the model appended after the first sentence
            concept = re.split(r"[.\n]", raw)[0].strip()
            return concept if concept else None
        except Exception as e:
            logger.warning(f"Latent {l} construction failed: {e}")
            return None

    # ── verification ──────────────────────────────────────────────────────

    def _predict_activation(self, l: int, concept: str, uid: int, true_level: int) -> int:
        """
        Ask LLM to predict activation level (0-10) for user uid / latent l.
        Returns predicted integer level; -1 on parse failure.
        """
        llm = self._get_llm()
        pred_iid = int(self.predicted_items[uid])
        test_seq, test_last = self._seq_for_user(uid, pred_iid, true_level)

        ex_seq, ex_expl, ex_last, ex_activ = self._example_for_dataset(self.item_category)

        messages = _build_verification_prompt(
            latent_id          = l,
            concept            = concept,
            test_sequence      = test_seq,
            test_last_item     = test_last,
            item_category      = self.item_category,
            item_name          = self.item_name,
            example_concept    = ex_expl,
            example_sequence   = ex_seq,
            example_last_item  = ex_last,
            example_activation = ex_activ,
        )
        try:
            raw = llm.chat(messages, max_new_tokens=8)
            numbers = re.findall(r"\d+", raw)
            if numbers:
                return int(numbers[0])
        except Exception as e:
            logger.warning(f"Latent {l} verify failed: {e}")
        return -1

    def verify_concept(
        self,
        l:           int,
        concept:     str,
        pos_users:   List[int],
        neg_users:   List[int],
    ) -> float:
        """
        Compute confidence score = accuracy on pos+neg test cases (eq. 4).
        Positive: predicted level > VERIFY_THRESHOLD → hit.
        Negative: predicted level <= VERIFY_THRESHOLD → hit.
        """
        hits = 0
        total = 0

        for uid in pos_users:
            true_level = int(self.act_levels[uid, l])
            pred = self._predict_activation(l, concept, uid, true_level)
            if pred >= 0:
                hits  += int(pred > VERIFY_THRESHOLD)
                total += 1
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        for uid in neg_users:
            pred = self._predict_activation(l, concept, uid, 0)
            if pred >= 0:
                hits  += int(pred <= VERIFY_THRESHOLD)
                total += 1
            if self.sleep_s > 0:
                time.sleep(self.sleep_s)

        return hits / total if total > 0 else 0.0

    # ── full pipeline ─────────────────────────────────────────────────────

    def run(self) -> Tuple[Dict[int, Dict], Dict[int, float]]:
        """
        Iterate over all latents: select cases → construct concept → verify.

        Returns:
            concepts    : {latent_id: {concept, construct_users, verify_pos, verify_neg}}
            conf_scores : {latent_id: confidence_score ∈ [0,1]}
        """
        n_latents   = self.act_levels.shape[1]
        concepts    : Dict[int, Dict]  = {}
        conf_scores : Dict[int, float] = {}

        logger.info(f"[LLM] Interpreting {n_latents} latents …")

        for l in range(n_latents):
            construct, v_pos, v_neg = self._select_cases(l)
            if construct is None:
                continue

            concept = self.construct_concept(l, construct)
            if concept is None:
                continue

            confidence = self.verify_concept(l, concept, v_pos, v_neg)

            concepts[l]    = {
                "concept":         concept,
                "construct_users": construct,
                "verify_pos":      v_pos,
                "verify_neg":      v_neg,
            }
            conf_scores[l] = confidence

            if l % 50 == 0:
                verified = sum(1 for s in conf_scores.values() if s >= CONFIDENCE_THRESH)
                logger.info(
                    f"[LLM] latent {l}/{n_latents}  "
                    f"verified(P≥{CONFIDENCE_THRESH})={verified}"
                )

        logger.info(
            f"[LLM] Done. Concepts={len(concepts)}  "
            f"Verified={sum(1 for s in conf_scores.values() if s >= CONFIDENCE_THRESH)}"
        )
        return concepts, conf_scores
