# 16 — Long Short-Term Memory Networks (LSTMs)

> **Week 2 · Topic 16** · The architectural fix for everything RNNs got wrong — a dedicated long-term memory highway with explicit gates.

---

## The core idea

LSTMs solve the RNN vanishing gradient problem with one architectural insight: add a second memory stream — the **cell state** — that flows through the network via **addition** rather than multiplication. Gradients flow through addition unchanged, creating a highway across hundreds of timesteps. Three gates (forget, input, output) explicitly control what gets written to, kept in, and read from this long-term memory.

---

## Do LSTMs use semantic vectors?

Yes — but they consume them, not create them.

LSTMs take **word embeddings** as input xₜ at each timestep. Word embeddings are dense vectors encoding semantic meaning — "king" and "queen" are close in embedding space. These are the semantic vectors.

```
Input xₜ          → word embedding (semantic vector — what this word means)
Hidden state hₜ   → sequential context (what has happened so far in the sequence)
Cell state Cₜ     → long-term sequential memory (important things to remember)
```

The hidden state and cell state are contextual — not semantic word vectors. By the time the LSTM reaches word 50, the original embedding of word 1 has passed through 49 compression steps and may be distorted. This is exactly what Transformer attention fixes — it preserves and directly accesses every word's original embedding at every step.

---

## The jam session analogy

You are 30 minutes into an epic jam. A vanilla RNN musician has only short-term working memory — anything from 10 minutes ago is gone. An LSTM musician carries two things:

**A whiteboard (hidden state hₜ)** — short-term working memory. What is relevant right now for the next note. Updated aggressively at every step.

**A notebook (cell state Cₜ)** — long-term memory. Deliberately managed with explicit decisions:
- **Forget gate:** "That rhythm from minute 3 — cross it out. We've moved on."
- **Input gate:** "This new chord progression is the heart of the jam — write it down."
- **Output gate:** "Right now I only need the key and emotional arc — not every detail in the notebook."

The notebook is the breakthrough. Unlike the whiteboard (rewritten every step), the notebook persists. And crucially — updates are **additive** (write new, cross out old) not **multiplicative** (overwrite everything). This is what lets gradients flow backwards unchanged across hundreds of timesteps.

---

## Two memory streams

```
Cell state Cₜ ————————————————————————————————————→  (long-term highway)
                ↕ gates         ↕ gates        ↕ gates
            [LSTM t=1]      [LSTM t=2]      [LSTM t=3]
                ↕ h              ↕ h             ↕ h
Hidden state hₜ →→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→→  (short-term, used for output)
```

**Cell state Cₜ** — long-term memory highway. Flows across timesteps with only minor modifications — addition and element-wise multiplication by gate values. Gradients flow through it nearly unchanged over hundreds of steps.

**Hidden state hₜ** — short-term working memory. Recomputed from scratch at every step. Used as output at each timestep and passed to the next cell. Same role as the RNN hidden state but now informed by long-term memory.

---

## The three gates

### Forget Gate — what to erase

```
fₜ = sigmoid(Wf · [hₜ₋₁, xₜ] + bf)
```

Output: 0 = completely forget, 1 = keep completely. Values between = partial retention.

Takes the previous hidden state and current input → sigmoid → produces a value per cell state element controlling how much of the old memory to keep.

**Notebook analogy:** "That rhythm pattern from 10 minutes ago — we have completely moved past it. Cross it out."

**Language example:** when a new subject starts ("The new guitarist..."), the LSTM forgets the gender of the previous subject — no longer relevant for pronoun agreement.

---

### Input Gate — what to write

```
iₜ = sigmoid(Wᵢ · [hₜ₋₁, xₜ] + bᵢ)   ← how much to write
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)      ← what to write
```

Two parts: iₜ decides HOW MUCH of the new candidate memory C̃ₜ to actually write. C̃ₜ is what the network proposes to write — a new candidate based on current input and hidden state.

**Notebook analogy:** "This new chord progression is the heart of where we are going — write it down carefully."

**Language example:** when a new subject is introduced ("She"), write the gender (female) to long-term memory for future pronoun agreement.

---

### Output Gate — what to read

```
oₜ = sigmoid(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ × tanh(Cₜ)
```

Decides which parts of the cell state are relevant right now and exposes them as the hidden state output hₜ.

**Notebook analogy:** "Right now for my next note I only need the key and emotional arc — not every single detail."

**Language example:** when generating a verb, read the subject from memory to ensure grammatical agreement — ignore the stored setting and tone for now.

---

### Cell state update — the full equation

```
Cₜ = fₜ × Cₜ₋₁  +  iₜ × C̃ₜ
     ↑ keep old      ↑ add new
```

This is **addition** — not repeated matrix multiplication. The gradient flows backwards through this operation with derivative ≈ 1 when fₜ ≈ 1 (preserved memories). No weight matrix multiplication, no tanh squashing — a direct highway.

---

## Why LSTMs solve vanishing gradients — the precise reason

**RNN gradient path:**
```
gradient at t=1 = gradient at t=T × (Wₕ × tanh')^(T-1)
                → shrinks to near zero over long sequences
```

**LSTM gradient path through cell state:**
```
∂Cₜ/∂Cₜ₋₁ = fₜ
```

The gradient of the cell state with respect to the previous cell state is simply the forget gate value. When fₜ ≈ 1 (keep this memory), the gradient flows through unchanged — no multiplication by a weight matrix, no squashing through tanh.

The network learns to set fₜ ≈ 1 for memories it wants to preserve — creating a direct gradient highway across any number of timesteps. **This is the fundamental breakthrough of LSTMs — not the gates themselves, but the additive cell state that keeps the gradient highway open.**

---

## Why Transformers were still needed

LSTMs solved vanishing gradients but introduced new problems:

**Problem 1 — Sequential bottleneck (the biggest one):**
LSTMs process tokens one at a time — step 2 cannot start until step 1 finishes. Modern GPUs are built for parallelism — LSTMs use maybe 10% of available compute. Transformers process all tokens simultaneously — full GPU utilisation. This gives Transformers a 10–100x training speed advantage at scale.

**Problem 2 — Fixed-size compression bottleneck:**
Even with the cell state, everything must be compressed into fixed-size vectors before the decoder starts. For translating a 500-word document, the encoder's final cell state must carry all context — information loss is inevitable. Transformers maintain the full representation of every token and allow direct token-to-token attention at any distance.

**Problem 3 — Semantic vector degradation:**
Word embeddings (semantic vectors) fed as input xₜ get progressively transformed through compression. By the time the LSTM reaches word 50, word 1's original semantic vector has passed through 49 compression steps. Transformer attention looks directly at every word's original embedding at every step — semantic information is preserved perfectly.

```
LSTM:       word_1_embedding → compress → compress → compress → (degraded)
Transformer: word_1_embedding ——————————————————————————→ directly attended to
```

---

## GRU — the simplified LSTM

The Gated Recurrent Unit merges the forget and input gates into one "update gate" and eliminates the separate cell state — combining it with the hidden state. Two gates instead of three, one memory stream instead of two.

Nearly identical performance to LSTM on most tasks, faster to train. Worth knowing for interviews.

---

## RNN vs LSTM vs Transformer

| Property | RNN | LSTM | Transformer |
|---|---|---|---|
| **Memory** | Hidden state only | Hidden + cell state | Full sequence (attention) |
| **Long-range dependencies** | Fails (~10–20 steps) | Good (~100s of steps) | Direct (any length) |
| **Processing** | Sequential | Sequential | Parallel |
| **Training speed** | Slow | Slow | Fast (GPU parallel) |
| **Semantic preservation** | Degrades fast | Degrades slowly | Preserved (direct attention) |
| **Still used?** | Rarely | Time series, streaming | Everything language-related |

---

## ⚠️ Common confusions

**Confusion: the cell state and hidden state are the same thing.**
They serve different roles. The cell state is long-term memory — updated additively, flows through the network as a gradient highway. The hidden state is short-term working memory — recomputed from scratch every step, used for output. The cell state is internal to the LSTM. The hidden state is what gets passed to the next layer and used for predictions.

**Confusion: LSTMs fully solved the sequence memory problem.**
LSTMs dramatically improved long-range memory — from ~20 steps (RNN) to hundreds of steps. But the cell state is still a fixed-size vector that can be overwritten. For very long sequences (1000+ tokens) and the need to preserve exact semantic vectors, Transformers are the real solution.

**Confusion: forget gate = 0 means the LSTM has forgotten the whole sequence.**
The forget gate acts element-wise on the cell state vector — it can forget specific dimensions while preserving others. It does not reset the entire memory. Different parts of the cell state can have different forget gate values simultaneously.

**Confusion: LSTMs use semantic vectors for their hidden/cell states.**
LSTMs consume semantic word embeddings as input xₜ. Their hidden and cell states are sequential context vectors — not semantic word vectors. The distinction matters: the hidden state encodes "what has happened so far" not "what this word means."

---

## Interview-ready summary

> "LSTMs upgrade RNNs with two memory streams — a short-term hidden state and a long-term cell state highway. The cell state updates additively (Cₜ = fₜ × Cₜ₋₁ + iₜ × C̃ₜ) rather than multiplicatively, allowing gradients to flow backwards unchanged across hundreds of timesteps. Three gates control memory: the forget gate decides what to erase (sigmoid → 0 to 1 per element), the input gate decides what new information to write, and the output gate decides what to expose as the current hidden state. LSTMs still failed to replace Transformers because they are fundamentally sequential (cannot parallelise training, wastes GPU), require compressing all context into fixed-size vectors (information loss), and still degrade semantic information over very long sequences where Transformers preserve it via direct attention."

---

## Resources
- **Udemy:** Deep Learning A-Z — Kirill Eremenko (Part 3: RNNs — LSTM section)
- **YouTube:** StatQuest — "Long Short-Term Memory (LSTM), Clearly Explained"
- **YouTube:** 3Blue1Brown — after completing Transformers topic for comparison

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
