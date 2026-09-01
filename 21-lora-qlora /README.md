# 21 — LoRA & QLoRA

> **Week 3 · Topic 21** · Fine-tuning a 70B model on a single GPU — the technique that democratised LLM adaptation.

---

## The core idea

LoRA freezes all original model weights and adds tiny trainable low-rank matrices alongside them — updating only those tiny matrices during fine-tuning. Trainable parameters drop from billions to millions (0.06%–0.5%) while achieving near-identical results to full fine-tuning. QLoRA extends this by quantising the frozen base model to 4-bit, enabling fine-tuning of 70B models on consumer hardware.

---

## The problem LoRA solves

Full fine-tuning a 7B model requires ~112GB GPU memory (16 bytes per parameter: weights + gradients + Adam optimiser states). A 70B model needs ~1.1TB. Most teams cannot access this.

LoRA's insight: you don't need to update all the weights. Most of the adaptation can be captured by training a tiny fraction of parameters — specifically by learning small low-rank matrices that represent the weight update, rather than modifying the original weights at all.

---

## The sticky note analogy

Your band has been playing together for 10 years — thousands of songs, complex arrangements, deep musical intuition. That is the pre-trained model.

**Full fine-tuning** is like rewriting every musician's personal notes, habits, and muscle memory — adjusting every tiny behaviour across the whole band. Insanely expensive, risks overwriting things that work.

**LoRA** is like giving each musician a tiny sticky note to attach to their instrument. The note says: "when playing flamenco, modify your technique this specific way." The note is tiny. The original playing style is completely untouched. But the modification captures everything needed.

At training time: only the sticky notes are written. The original musicians (weights) are frozen.
At inference time: original playing + sticky note modification = the fine-tuned performance.

---

## The low-rank hypothesis — why LoRA works

When researchers studied what changes in model weights during fine-tuning, they found something surprising: **weight updates ΔW have low intrinsic rank.**

A 4096×4096 weight matrix could theoretically have rank 4096 — meaning 4096 truly independent directions of information. But weight updates during fine-tuning were found to have effective rank of just 1–64. The vast majority of adaptation lives in a very low-dimensional subspace.

If ΔW has low rank, it can be closely approximated by two small matrices:

```
ΔW ≈ B × A   where B is (d × r) and A is (r × d),   r << d
```

For d=4096 and r=8:
```
Full ΔW:  4096 × 4096 = 16,777,216 parameters
LoRA A:   8    × 4096 = 32,768 parameters
LoRA B:   4096 × 8    = 32,768 parameters
Total:    65,536 parameters   ← 99.6% reduction
```

---

## How LoRA works — the forward pass

```
Input x
  ↓         ↓ (parallel path)
W·x         A·x → B·(A·x)
(frozen)    (A and B are trainable)
  ↓         ↓
  └────+────┘
       ↓
  Output = W·x + B·A·x = (W + BA)·x
```

**W** — original pre-trained weight matrix. Completely frozen. Never updated. No gradient computed for it.

**A** — small matrix (r×d). Randomly initialised. Trained during fine-tuning.

**B** — small matrix (d×r). **Initialised to zeros** — so BA=0 at start. Model begins identical to base model. Trained during fine-tuning.

**Why B initialised to zeros:** at the start of fine-tuning you want the model to behave exactly like the base model. If B=0 then BA=0 and output = Wx — identical to original. If both A and B were randomly initialised, BA would be a random non-zero matrix from the very first forward pass, adding random noise to every layer simultaneously — catastrophic instability. Zero initialisation of B ensures training starts from the known-good baseline. A provides gradient diversity; B learns the meaningful adaptation.

Only A and B have gradients. Only A and B are stored in the optimiser. Only A and B require training memory.

---

## Where LoRA adapters are applied

LoRA adapters can be applied to any weight matrix. In practice they are most commonly applied to the attention projection matrices:

```
Query projection:  Wq
Key projection:    Wk
Value projection:  Wv
Output projection: Wo
```

Some implementations also apply to FFN layers. The choice of which matrices to adapt is a hyperparameter — more matrices = more parameters = better quality but higher memory.

---

## The rank hyperparameter r

| Rank | Trainable params (7B) | Use when |
|---|---|---|
| r=1–4 | ~1–2M | Minimal adaptation — style/format only |
| r=8–16 | ~4–8M | Standard sweet spot — most instruction tuning |
| r=32–64 | ~16–32M | Complex domain adaptation |
| r=128+ | ~64M+ | Approaching full fine-tuning territory |

**Rule of thumb:** start with r=8. If quality is insufficient try r=16 or r=32. Most instruction fine-tuning tasks are solved at r=8–16. The low-rank hypothesis holds strongly in practice.

Higher r = more expressive = more memory. The goal is the lowest r that achieves acceptable quality.

---

## QLoRA — quantisation + LoRA

QLoRA combines two techniques to enable fine-tuning on consumer hardware:

**1. NF4 quantisation of the base model:**
The frozen base model weights are loaded in 4-bit precision (NF4 — NormalFloat4) instead of 16-bit or 32-bit:
```
7B model in FP16:  ~14GB
7B model in NF4:   ~3.5GB   ← 4x compression
```

**2. LoRA adapters in BF16:**
The small A and B matrices are trained in BF16 (16-bit brain float). Gradients flow through the quantised weights via double quantisation — the model learns despite the base weights being in low precision.

**Result:**
```
Full fine-tuning (7B):    ~112GB   → multiple A100s
Standard LoRA (7B):       ~16–20GB → 1–2 A100s
QLoRA (7B):               ~6–8GB   → single consumer GPU (RTX 3080/4080)
QLoRA (70B):              ~40–48GB → 2 consumer GPUs
```

QLoRA is what democratised LLM fine-tuning in 2023 — researchers without expensive clusters could suddenly fine-tune frontier-scale models.

**When to choose QLoRA over LoRA:** only have consumer-grade GPUs, fine-tuning large models (13B+), cost/accessibility matters more than maximum quality. QLoRA introduces a small quality degradation from quantisation — in production with proper infrastructure, standard LoRA is preferred.

---

## LoRA adapters — merge or swap

### Merging — for single-task deployment

After training, merge the LoRA weights back into the original:
```
W_new = W + B×A
```
Result is a standard model with no inference overhead — runs exactly as fast as the original. Use when you have one primary task and want maximum inference speed.

### Swapping — for multi-task deployment

Keep the base model frozen. Train multiple LoRA adapters — one per task. At inference time, load the base model once and swap the tiny adapter (~10–100MB) based on the incoming task.

```
Base model (7B) — loaded once, stays in memory
  + adapter_support.safetensors    (~20MB)  → customer support requests
  + adapter_code.safetensors       (~20MB)  → code generation requests
  + adapter_summarise.safetensors  (~20MB)  → summarisation requests
  + adapter_translate.safetensors  (~20MB)  → translation requests
```

One base model, many specialisations. Enormously cost-efficient at scale — five tasks served from one model instead of five separate models.

---

## LoRA vs full fine-tuning

| Property | Full fine-tuning | LoRA |
|---|---|---|
| **Parameters updated** | All (7B for 7B model) | ~4–40M (0.06%–0.5%) |
| **Memory (7B)** | ~112GB | ~16–20GB (~6–8GB with QLoRA) |
| **Catastrophic forgetting** | High risk | Low risk — original weights untouched |
| **Quality** | Maximum | Near-identical for most tasks |
| **Multiple tasks** | Separate models per task | One base + swap adapters |
| **Training speed** | Slower | Faster — fewer gradient computations |
| **Default choice?** | Only when max quality needed + compute available | Yes — default for most fine-tuning |

---

## ⚠️ Common confusions

**Confusion: LoRA trains a completely separate model.**
LoRA trains small adapter matrices that are added on top of the frozen original model. The original model weights are never modified. At inference time the output is W·x + BA·x — the original model output plus a small learned correction. It is not a separate model; it is a modification layer on top of the original.

**Confusion: higher rank always means better quality.**
Diminishing returns set in quickly. Most instruction fine-tuning tasks see no quality improvement beyond r=16–32. Higher rank increases memory and training time without proportional quality gains. Start with r=8 and only increase if validation metrics are insufficient.

**Confusion: QLoRA significantly degrades model quality.**
In practice, QLoRA (NF4 quantisation + LoRA) achieves quality comparable to full 16-bit LoRA on most tasks. The quantisation error is small and the LoRA adapter compensates for much of it during training. For production deployments requiring maximum quality, standard LoRA in 16-bit is preferred — but for most research and many production cases, QLoRA quality is acceptable.

**Confusion: LoRA only applies to attention weights.**
LoRA can be applied to any weight matrix — attention projections (Wq, Wk, Wv, Wo), FFN layers, embedding matrices. In practice attention projections are most commonly targeted because they capture the most adaptation per parameter. The choice of which layers to apply LoRA to is a hyperparameter called target_modules.

---

## Interview-ready summary

> "LoRA exploits the low-rank hypothesis — weight updates during fine-tuning have low intrinsic rank and can be approximated by multiplying two small matrices A (r×d) and B (d×r) rather than storing the full update ΔW. The original model weights are frozen; only A and B are trained. B is initialised to zero so training starts identical to the base model. This reduces trainable parameters from billions to millions (0.06%–0.5%) and memory from 112GB to 16–20GB for a 7B model. Rank r controls the trade-off between expressiveness and efficiency — r=8–16 is the standard sweet spot. QLoRA additionally quantises the frozen base model to 4-bit (NF4), enabling fine-tuning on consumer GPUs. Multiple LoRA adapters can be trained for different tasks and swapped at inference time over one shared base model — one base model, many specialisations. LoRA is the default fine-tuning choice for virtually all practical scenarios."

---

## Resources
- **Udemy:** LLM Engineering — Ed Donner (LoRA and fine-tuning sections)
- **Paper:** "LoRA: Low-Rank Adaptation of Large Language Models" — Hu et al. 2021
- **Paper:** "QLoRA: Efficient Finetuning of Quantized LLMs" — Dettmers et al. 2023
- **Library:** Hugging Face PEFT — the standard LoRA implementation

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
