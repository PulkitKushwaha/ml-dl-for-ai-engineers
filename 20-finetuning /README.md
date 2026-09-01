# 20 — Fine-tuning

> **Week 3 · Topic 20** · Taking a general-purpose LLM and making it exceptional at one specific thing — without training from scratch.

---

## The core idea

Fine-tuning continues training a pre-trained model on a smaller, task-specific dataset — updating its weights so it becomes exceptional at one thing while retaining most of what it learned during pre-training. It is not retraining from scratch. It is focused immersion on top of existing capability.

---

## The adaptation spectrum

```
Prompting          RAG              Fine-tuning        Pre-training
────────────────────────────────────────────────────────────────────
No training        No training      Updates weights    Trains from scratch
Cheapest           Adds knowledge   Changes behaviour  Most expensive
Hours to iterate   Days to build    Days to weeks      Months + $millions
```

Fine-tuning sits between RAG and pre-training — starts from an existing model, updates weights on new data, orders of magnitude cheaper than pre-training but far more powerful than prompting alone.

---

## The session musician analogy

A session musician is one of the best guitarists in the world — jazz, blues, rock, country, classical. That is a pre-trained LLM: trained on the internet, capable of almost anything.

A record label hires them exclusively for flamenco. They do not start from scratch. They take everything they know and spend a focused few weeks deeply immersed in flamenco. After that focused period they are exceptional at it.

That focused immersion is fine-tuning. The model does not forget how to play guitar — it builds specialisation on top of existing capability.

**The risk:** lock them in a room with nothing but flamenco for six months and they might start forgetting how to play blues. That is catastrophic forgetting — fine-tuning so hard on narrow data that general capability degrades.

---

## What fine-tuning actually does — mechanically

Pre-training trains on trillions of tokens — massive compute, months of GPU time, billions of dollars. The result is a base model with rich general knowledge but no specific behaviour.

Fine-tuning takes that base model and continues gradient descent on a small curated dataset. Same backpropagation and weight update process — just with new data and far fewer steps.

**What changes:** weights shift slightly to better predict fine-tuning data. Patterns in fine-tuning data get reinforced. Patterns absent from it may weaken.

**What doesn't change:** the architecture, vocabulary, tokeniser. Fine-tuning is purely a weight update process — same model, adjusted weights.

---

## Types of fine-tuning

### Continued Pre-training — adds knowledge

Train the base model on new raw text using the same next-token-prediction objective as pre-training, but on domain-specific data. Used when the model lacks knowledge of a specific domain entirely.

**Example:** a medical LLM trained on millions of clinical notes, research papers, and drug databases. LLaMA knows medicine from the internet — continued pre-training immerses it in clinical language and medical reasoning patterns.

**Data needed:** large — typically millions to billions of tokens. Expensive. Use only when the domain is truly underrepresented in the base model's training data.

---

### Instruction Fine-tuning (SFT — Supervised Fine-tuning) — changes behaviour

Train on (instruction, response) pairs — teaching the model to follow instructions, use a specific format, and adopt a particular tone or persona. The most common fine-tuning type in production.

**Example:** a customer support bot fine-tuned on thousands of (customer query, ideal response) pairs. The model learns to respond formally, stay on-topic, follow escalation procedures, and avoid certain topics.

**Data format:**
```json
{
  "instruction": "Classify this review as positive or negative",
  "input": "The guitar sounded amazing but delivery was slow",
  "output": "Mixed — positive product sentiment, negative delivery sentiment"
}
```

**Data needed:** surprisingly small. 1,000–10,000 high-quality examples often sufficient. Quality matters far more than quantity.

---

### Full Fine-tuning — updates all weights

Updates every single weight in the model — all layers, all attention heads, all FFN weights. Maximum expressive power but enormous compute and memory requirements.

**The memory problem:** a 7B parameter model needs:
```
Weights:          4 bytes × 7B params = 28GB   (FP32)
Gradients:        4 bytes × 7B params = 28GB
Optimiser states: 8 bytes × 7B params = 56GB   (Adam: momentum + variance)
─────────────────────────────────────────────────────
Total:            16 bytes × 7B params = 112GB
```

A 70B model requires ~1.1TB total. Essentially impossible without a large cluster. This is exactly why LoRA was invented — Topic 21.

---

## Catastrophic forgetting — the main risk

When a model is fine-tuned aggressively on narrow data, it can lose general capabilities. The weights shift so far toward the fine-tuning distribution that previously learned patterns are overwritten.

**Concrete example:** fine-tune GPT on 100,000 formal legal writing examples. The model becomes excellent at legal prose but struggles with casual conversation, code, and creative writing — patterns present in pre-training but absent from fine-tuning.

**Fixes:**
1. Mix fine-tuning data with general data — 10–20% general, 80–90% domain-specific
2. Use a low learning rate — small weight updates preserve more pre-training knowledge
3. Fine-tune fewer layers — update only later layers, freeze earlier ones
4. Use LoRA — updates only a tiny fraction of weights, minimising forgetting by design

---

## Data requirements — quality beats quantity

### What makes fine-tuning data good

- **Diverse coverage** of the target task — not 10,000 near-identical examples
- **Clean and consistent format** — the model learns format from data
- **Correct outputs** — mislabelled data directly teaches wrong behaviour
- **Representative edge cases** — failure modes must appear in training
- **Human-verified** — especially for instruction tuning

### Rough data size guidelines

| Goal | Data needed | Notes |
|---|---|---|
| Style / format / persona | 100–1,000 examples | Model already knows how to write — redirecting it |
| Task-specific behaviour | 1,000–10,000 examples | Teaching reasoning patterns or domain responses |
| Domain knowledge | 10,000–100,000+ examples | Teaching genuinely new facts |
| Continued pre-training | 100M+ tokens | Treating knowledge gaps as a pre-training problem |

---

## When to fine-tune — the decision conditions

### ✅ Fine-tune when

- **Consistent format/style is critical** — legal, medical, brand voice requires exact output format
- **Task requires specific reasoning patterns** not present in base model behaviour
- **Latency matters** — a smaller fine-tuned model is faster than a larger prompted model
- **Cost at scale** — one fine-tuned call is cheaper than a long system prompt × millions of calls
- **Privacy** — sensitive data cannot go in prompts to external APIs; fine-tune on-premise instead

### ❌ Don't fine-tune when

- **You just need to add factual knowledge** — use RAG instead; fine-tuning does not reliably inject facts
- **You don't have high-quality labelled examples** — bad fine-tuning data makes the model worse
- **A good system prompt solves it** — always try prompting first; it is 100x cheaper to iterate
- **You need the model to stay current** — fine-tuned models freeze knowledge at training time
- **You are still iterating on the task definition** — fine-tuning is expensive to redo repeatedly

---

## The compute reality — why this matters

| Model size | Full fine-tuning memory | GPUs needed |
|---|---|---|
| 7B | ~112GB | 2–4 × A100 (80GB) |
| 13B | ~200GB | 4–8 × A100 |
| 70B | ~1.1TB | Large cluster |
| GPT-4 scale | Estimated $100M+ | Not practical |

Memory comes from three simultaneous requirements: weights + gradients + optimiser states = 16 bytes per parameter in FP32 with Adam.

This is the exact problem LoRA solves — by updating only a tiny fraction of parameters, memory requirements drop from 112GB to ~16GB or less for a 7B model. Topic 21.

---

## ⚠️ Common confusions

**Confusion: fine-tuning the model to add new knowledge.**
Fine-tuning is for changing behaviour, style, and task-specific reasoning — not for injecting facts reliably. Models can still hallucinate facts even after fine-tuning on documents containing those facts. For knowledge injection use RAG — it retrieves and grounds answers in source documents at inference time.

**Confusion: more fine-tuning data always means better results.**
Quality and diversity beat raw volume. 1,000 carefully curated, diverse, correctly labelled examples consistently outperforms 100,000 noisy, repetitive, or mislabelled ones. The model learns format, style, and reasoning patterns from the distribution of examples — bad distribution = bad fine-tuning.

**Confusion: fine-tuning always causes catastrophic forgetting.**
Catastrophic forgetting is a risk, not an inevitability. Low learning rates, data mixing (10–20% general data), layer freezing, and LoRA all dramatically reduce the risk. A well-executed instruction fine-tune on 5,000 examples rarely causes significant forgetting.

**Confusion: fine-tuning and RLHF are the same thing.**
Instruction fine-tuning (SFT) teaches the model to follow instructions by training on (instruction, response) pairs. RLHF is a separate subsequent step that uses human preference data and reinforcement learning to make the model more helpful, harmless, and honest. SFT comes first; RLHF comes after. Topic 22.

---

## Interview-ready summary

> "Fine-tuning continues training a pre-trained LLM on a small curated dataset — the same gradient descent and backpropagation process, just with new data and far fewer steps. It changes behaviour, style, and task-specific reasoning rather than injecting new knowledge (that's RAG). Three types: continued pre-training for domain knowledge gaps (millions of tokens), instruction fine-tuning for behaviour/format changes (1,000–10,000 high-quality examples), and full fine-tuning which updates all weights but requires enormous memory (16 bytes per parameter = 112GB for a 7B model). The main risks are catastrophic forgetting (fixed with data mixing, low learning rate, LoRA) and data quality (1,000 clean examples beats 100,000 noisy ones). Fine-tune when consistent format, privacy, latency, or cost at scale matters. Use RAG when the problem is factual knowledge retrieval. Use prompting first — always."

---

## Resources
- **Udemy:** LLM Engineering — Ed Donner (fine-tuning sections)
- **YouTube:** Andrej Karpathy — "State of GPT" (fine-tuning pipeline explained)
- **Paper:** "Training language models to follow instructions with human feedback" — OpenAI InstructGPT paper

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
