import json, gc, os, re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
# TOP_ITEMS_PATH   = "top50item_amazon-grocery_LightGCN-Apr-08-2026_23-24-12-neuron_top_items.json"
TOP_ITEMS_PATH  = "/kaggle/input/datasets/huysonnguyen/top50item-per-neuron-sae/top50item_amazon-grocery_LightGCN-Apr-08-2026_23-24-12-neuron_top_items.json"
OUTPUT_PATH      = "grocery_Llama3-8B_LightGCN_neuron_labels.json"
ITEM_CATEGORY    = "grocery and gourmet"
TOP_K_FOR_PROMPT = 10
MAX_NEW_TOKENS   = 80
SAVE_EVERY       = 10
LLM_BACKEND      = "huggingface"
LLM_MODEL        = "meta-llama/Llama-3.1-8B-Instruct"
HF_BATCH_SIZE    = 1          # 1 prompt at a time to avoid OOM
USE_4BIT         = False       # bitsandbytes 4-bit quantization

PRIORITY_FIELDS  = ["title", "category", "brand", "price", "popularity_rank"]


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder  (fixed: explicit format + few-shot example)
# ═══════════════════════════════════════════════════════════════════════════════

def format_item(item: Dict, max_fields: int = 5) -> str:
    parts, used = [], 0
    for field in PRIORITY_FIELDS:
        if field in item and used < max_fields:
            parts.append(f'{field}="{item[field]}"')
            used += 1
    return f"  {int(item['rank']):2d}. {' | '.join(parts)}  [activation={item['activation']:.3f}]"


def build_prompt(neuron_id: int, items: List[Dict], category: str) -> List[Dict]:
    system = (
        f"You analyse neurons in recommender system for {category}. "
        f"A neuron activates for items sharing a specific characteristic. "
        f"Given the top-activated {category} items for each neuron, output EXACTLY 3-5 characteristics with confidence scores (0-10) covering the most prominent shared features (brand, types, dietary attribute, price tier, cuisine origin, ingredient, popularity, etc. of {category} items.)"
        # f"Cover features like: brand, types, dietary attribute, price tier, cuisine origin, ingredient, popularity, etc. of {category} items. "
        f"Rules: (1) No numbering. (2) No preamble. (3) One line only. "
        f"(4) Use consistent terms, never synonyms for the same concept. "
        f"(5) Higher-activation items matter more.\n\n"
        f"Format: ConceptA (score), ConceptB (score), ConceptC (score)\n"
        f"Example: Sugar-Free (9), Beverage Syrup (8), Premium Brand (6), Low Calorie (5)"
    )
    items_block = "\n".join(format_item(it) for it in items[:TOP_K_FOR_PROMPT])
    user = (
        f"Neuron {neuron_id} — top activated {category}:\n"
        f"{items_block}\n\n"
        f"Label (3-5 mutual characteristics with confidence scores):"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# ═══════════════════════════════════════════════════════════════════════════════
# Label cleaner
# ═══════════════════════════════════════════════════════════════════════════════

def clean_label(raw: str) -> str:
    if not raw:
        return "unlabelled"
    text = str(raw).strip()
    # Strip <think> blocks (DeepSeek / QwQ style)
    while "<think>" in text and "</think>" in text:
        s, e = text.find("<think>"), text.find("</think>")
        if e > s:
            text = (text[:s] + text[e + 8:]).strip()
        else:
            break
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and ln.strip() not in ("<think>", "</think>")]
    if not lines:
        return "unlabelled"
    label = lines[0].strip("\"'`")
    # Remove leading "Labels:" or numbering artefacts like "1. "
    label = re.sub(r'^(labels?:?\s*|[\d]+\.\s*)', '', label, flags=re.IGNORECASE).strip()
    return label if label else "unlabelled"


def parse_label_concepts(label: str) -> List[Tuple[str, float]]:
    """
    Parse 'ConceptA (9), ConceptB (7)' → [('ConceptA', 9.0), ('ConceptB', 7.0)].
    Used downstream for analysis.
    """
    concepts = []
    for match in re.finditer(r'([^,(]+?)\s*\(\s*(\d+(?:\.\d+)?)\s*\)', label):
        name  = match.group(1).strip().strip('"\'')
        score = float(match.group(2))
        if name:
            concepts.append((name, score))
    return concepts


# ═══════════════════════════════════════════════════════════════════════════════
# Memory helpers
# ═══════════════════════════════════════════════════════════════════════════════

def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def gpu_free_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1024 ** 3


# ═══════════════════════════════════════════════════════════════════════════════
# HuggingFace backend  (4-bit quantized, batch_size=1)
# ═══════════════════════════════════════════════════════════════════════════════

class HuggingFaceBackend:
    """
    Memory-optimised HF backend for Llama-3.1-8B on 2×T4 (15 GB each).

    Key changes vs original:
    - 4-bit NF4 quantisation via bitsandbytes  → ~5 GB vs ~16 GB at fp16
    - batch_size=1 (avoids KV-cache growth across long prompts)
    - generate() called directly instead of pipeline() for fine-grained control
    - explicit cache clear after every forward pass
    """

    def __init__(self, model_name: str, use_4bit: bool = True):
        from transformers import (AutoTokenizer, AutoModelForCausalLM,
                                  BitsAndBytesConfig)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        quant_cfg = None
        if use_4bit and self._device == "cuda":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            print("Loading model with 4-bit NF4 quantisation …")
        else:
            print("Loading model at fp16 …")

        self._tok = AutoTokenizer.from_pretrained(model_name)
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id
        self._tok.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
            torch_dtype=torch.float16 if not use_4bit else None,
        )
        self._model.eval()
        print(f"Model loaded. GPU free: {gpu_free_gb():.2f} GB")

    def label_one(self, messages: List[Dict]) -> str:
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(prompt, return_tensors="pt", truncation=True,
                           max_length=1024).to(self._device)

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._tok.pad_token_id,
                eos_token_id=self._tok.eos_token_id,
            )

        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        text = self._tok.decode(new_tokens, skip_special_tokens=True).strip()

        del inputs, output_ids, new_tokens
        clear_gpu()
        return text


class OpenAIBackend:
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model  = model

    def label_one(self, messages: List[Dict]) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages,
                max_tokens=MAX_NEW_TOKENS, temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[OpenAI error] {e}")
            return "unlabelled"


class AnthropicBackend:
    def __init__(self, model: str, api_key: Optional[str] = None):
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model

    def label_one(self, messages: List[Dict]) -> str:
        system    = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_msgs = [m for m in messages if m["role"] != "system"]
        try:
            resp = self._client.messages.create(
                model=self._model, max_tokens=MAX_NEW_TOKENS,
                system=system, messages=user_msgs,
            )
            return resp.content[0].text.strip()
        except Exception as e:
            print(f"[Anthropic error] {e}")
            return "unlabelled"


def make_backend():
    if LLM_BACKEND == "openai":
        return OpenAIBackend(LLM_MODEL, os.environ["OPENAI_API_KEY"])
    if LLM_BACKEND == "anthropic":
        return AnthropicBackend(LLM_MODEL)
    return HuggingFaceBackend(LLM_MODEL, use_4bit=USE_4BIT)


# ═══════════════════════════════════════════════════════════════════════════════
# Main labelling
# ═══════════════════════════════════════════════════════════════════════════════

def label_neurons(
    top_items_path: str = TOP_ITEMS_PATH,
    output_path: str    = OUTPUT_PATH,
    resume: bool        = True,
) -> Dict[str, str]:
    with open(top_items_path, encoding="utf-8") as f:
        neuron_top_items: Dict[str, List[Dict]] = json.load(f)

    existing: Dict[str, str] = {}
    if resume:
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            print(f"Resuming — {len(existing)}/{len(neuron_top_items)} already labelled.")
        except FileNotFoundError:
            pass

    llm     = make_backend()
    results = dict(existing)
    neurons = sorted(neuron_top_items.keys(), key=int)
    pending = [n for n in neurons if n not in results]
    print(f"Total: {len(neurons)} | Pending: {len(pending)}")

    for i, neuron_id in enumerate(pending, 1):
        items  = neuron_top_items[neuron_id]
        prompt = build_prompt(int(neuron_id), items, ITEM_CATEGORY)
        raw    = llm.label_one(prompt)
        label  = clean_label(raw)
        results[neuron_id] = label
        print(f"[{i:4d}/{len(pending)}] neuron {neuron_id:>4s} → {label}")

        if i % SAVE_EVERY == 0 or i == len(pending):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"  ✓ checkpoint saved ({len(results)} labels)")

    return results

if __name__ == "__main__":
    labels = label_neurons()
    print(f"\nDone. {len(labels)} neurons labelled.")