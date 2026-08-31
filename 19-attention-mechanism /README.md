# 19 — Attention Mechanism Deep Dive

> **Week 2 · Topic 19** · The mathematical heart of every modern LLM — computed step by step.

---

## The core idea

Attention is a mechanism that computes a new representation for each token by taking a weighted blend of all other tokens' representations — where the weights are determined by how relevant each token is to the current one. It replaces the sequential processing of RNNs with direct, parallel, distance-independent connections between any two tokens.

---

## Why attention was revolutionary

Before attention (2014–2017), sequence models had to compress entire input sequences into a fixed-size vector before decoding. For long sequences this was catastrophic — a 500-word document compressed into a 512-dimensional vector loses enormous amounts of information.

The original attention paper (Bahdanau, 2014) added attention to encoder-decoder RNNs — instead of only using the final encoder hidden state, the decoder could attend to all encoder hidden states at each decoding step. Performance on long sequences improved dramatically.

The 2017 "Attention Is All You Need" paper replaced the RNN entirely with attention — the Transformer. Attention was no longer an add-on but the entire architecture.

---

## Step-by-step attention computation

### Setup

Suppose we have 4 tokens: "The", "solo", "was", "epic"
Each token has been embedded to a 4-dimensional vector (simplified — real models use 512–12,288 dimensions):

```
Token embeddings (after positional encoding added):
"The"  = x₁ = [1.0, 0.2, -0.5, 0.8]
"solo" = x₂ = [0.3, 1.1, 0.4, -0.2]
"was"  = x₃ = [-0.1, 0.5, 0.9, 0.3]
"epic" = x₄ = [0.8, -0.3, 0.2, 1.2]
```

### Step 1 — Compute Q, K, V for every token

Multiply each token embedding by three separate learned weight matrices Wq, Wk, Wv:

```
For token "epic" (x₄):
Q₄ = Wq · x₄   ← "What is epic looking for in the other tokens?"
K₄ = Wk · x₄   ← "What does epic advertise it contains?"
V₄ = Wv · x₄   ← "What will epic contribute if someone attends to it?"
```

These weight matrices are learned during training — they are what the model learns during the training process.

### Step 2 — Compute attention scores

For each token, compute how much it should attend to every other token using the dot product of its Query with every Key:

```
Computing attention scores for "epic" (token 4):

score(epic → The)  = Q₄ · K₁ = dot product = 0.8
score(epic → solo) = Q₄ · K₂ = dot product = 2.1
score(epic → was)  = Q₄ · K₃ = dot product = 3.4
score(epic → epic) = Q₄ · K₄ = dot product = 4.2
```

### Step 3 — Scale by √dₖ

Divide all scores by the square root of the key dimension (dₖ = 4 in this example, √4 = 2):

```
Scaled scores:
score(epic → The)  = 0.8 / 2 = 0.40
score(epic → solo) = 2.1 / 2 = 1.05
score(epic → was)  = 3.4 / 2 = 1.70
score(epic → epic) = 4.2 / 2 = 2.10
```

**Why scale?** Without scaling, dot products grow with dimension size — in high-dimensional spaces (dₖ = 512 or 12,288) scores become very large, pushing the softmax into saturation regions where gradients are near zero. Scaling keeps scores in a reasonable range.

### Step 4 — Softmax to get attention weights

Convert scaled scores to probabilities (all sum to 1):

```
softmax([0.40, 1.05, 1.70, 2.10]):

exp(0.40) = 1.49
exp(1.05) = 2.86
exp(1.70) = 5.47
exp(2.10) = 8.17
sum = 17.99

Attention weights:
α(epic → The)  = 1.49 / 17.99 = 0.083  (8.3% attention)
α(epic → solo) = 2.86 / 17.99 = 0.159  (15.9% attention)
α(epic → was)  = 5.47 / 17.99 = 0.304  (30.4% attention)
α(epic → epic) = 8.17 / 17.99 = 0.454  (45.4% attention)
```

"epic" attends most to itself (45.4%), then "was" (30.4%), then "solo" (15.9%), then "The" (8.3%).

### Step 5 — Weighted sum of Values

The new representation of "epic" is a weighted blend of all tokens' Value vectors:

```
new_epic = α₁ × V₁  +  α₂ × V₂  +  α₃ × V₃  +  α₄ × V₄
         = 0.083 × V("The")
         + 0.159 × V("solo")
         + 0.304 × V("was")
         + 0.454 × V("epic")
```

The result is a new vector for "epic" that now encodes contextual information about the whole sentence — not just what "epic" means in isolation, but what it means in the context of "The solo was."

### The full formula

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

This is computed for all tokens simultaneously as a matrix operation — extremely efficient on GPUs.

---

## The attention matrix

For a sequence of n tokens, attention produces an n×n matrix where entry (i,j) is how much token i attends to token j:

```
Attention matrix for "The solo was epic":

              "The"  "solo"  "was"  "epic"
"The"    →  [0.55,  0.20,  0.15,  0.10]
"solo"   →  [0.12,  0.48,  0.22,  0.18]
"was"    →  [0.08,  0.25,  0.42,  0.25]
"epic"   →  [0.08,  0.16,  0.30,  0.45]  ← (computed above)
```

Each row sums to 1.0 (softmax output).

**Computational complexity:** O(n²) — every token attends to every other token. For n=1000 tokens: 1,000,000 attention scores computed. For n=100,000 tokens: 10 billion scores. This is why long context windows are computationally expensive.

---

## Causal (masked) attention — how decoders work

In decoder-only models (GPT, Claude), each token can only attend to previous tokens — not future ones. This is enforced by adding a mask before the softmax:

```
Masked attention scores for "epic" (position 4):

Before mask:
[0.40, 1.05, 1.70, 2.10]   ← scores for The, solo, was, epic

After causal mask (future tokens set to -∞):
[0.40, 1.05, 1.70, 2.10]   ← all past (positions 1-4 all visible at position 4)

For "was" (position 3), mask applied:
[0.40, 1.05, 1.70, -∞]   ← position 4 ("epic") is masked — future token
After softmax: [0.15, 0.32, 0.53, 0.00]  ← "epic" gets 0 attention weight
```

Setting future scores to -∞ before softmax makes exp(-∞) = 0 — those tokens get zero attention weight. The model genuinely cannot see future tokens.

This is the autoregressive constraint — generation must happen left-to-right, predicting each token having seen only what came before.

---

## Cross-attention — encoder-decoder connection

In encoder-decoder models (T5, BART), the decoder uses cross-attention to attend to the encoder's output:

```
Cross-attention:
  Q comes from the decoder's current state  ("What is the decoder looking for?")
  K and V come from the encoder's output    ("What did the encoder understand?")
```

This allows the decoder to ask "which parts of the encoded input are relevant to what I'm currently generating?" At each decoding step, the decoder looks at the full encoded input and extracts the relevant information.

Cross-attention is what makes encoder-decoder models so effective for translation — the decoder for the French output can directly attend to the relevant English words in the encoded input at every generation step.

---

## Multi-head attention — deep dive

### Why multiple heads?

A single attention head can only learn one relationship pattern at a time. Language has simultaneous multi-dimensional structure:
- Syntactic: which words are grammatically related
- Semantic: which words have related meanings
- Coreference: which words refer to the same entity
- Positional: which words are physically nearby

Multi-head attention runs h independent attention operations, each with its own Wq, Wk, Wv matrices. Each head specialises in different relationship types — this specialisation emerges from training, not design.

### The multi-head computation

```
For each head i (i = 1 to h):
  headᵢ = Attention(Q·Wqᵢ, K·Wkᵢ, V·Wvᵢ)

MultiHead = Concat(head₁, head₂, ..., headₕ) · Wₒ
```

If d_model = 512 and h = 8 heads:
- Each head uses d_k = d_v = 512/8 = 64 dimensions
- 8 heads run in parallel, each producing 64-dimensional outputs
- Concatenated: 8 × 64 = 512 dimensions → projected back to 512 via Wₒ

The total parameter count is the same as one large attention head — the multi-head structure is computationally equivalent but learns richer representations.

---

## Attention patterns that emerge

Research into what attention heads actually learn (by visualising attention matrices) has found:

**Early layers:**
- Heads attending to adjacent tokens (local syntactic structure)
- Heads attending to punctuation (sentence boundaries)
- Heads with broad, diffuse attention (gathering context)

**Middle layers:**
- Coreference resolution heads ("she" → "Alice")
- Syntactic dependency heads (verb → subject)
- Positional heads (attending to fixed relative positions)

**Late layers:**
- Task-specific patterns
- Long-range semantic relationships
- Abstract reasoning patterns

This hierarchy mirrors what we saw in CNNs — early layers detect simple patterns, deep layers detect complex ones.

---

## Attention vs other sequence mechanisms

| | RNN | LSTM | Self-Attention |
|---|---|---|---|
| **Connection type** | Sequential through hidden state | Sequential through cell state | Direct — any token to any token |
| **Distance** | Degrades with distance | Handles ~100s of steps | Identical regardless of distance |
| **Parallelism** | None | None | Full — all tokens simultaneously |
| **Complexity** | O(n) | O(n) | O(n²) memory |
| **Gradient path** | Multiplicative — vanishes | Additive cell state highway | Direct — no long chains |

---

## Efficient attention variants

The O(n²) complexity of standard attention becomes a bottleneck for very long sequences:

**Sparse attention (Longformer):** each token attends to a local window of nearby tokens plus a few global tokens. O(n) instead of O(n²) for local patterns.

**FlashAttention:** does not change the mathematical result but reorders the computation to be cache-friendly — 2–4x faster in practice with the same output. Standard in modern LLM training.

**Linear attention:** approximates the attention mechanism with O(n) complexity. Quality trade-off vs speed gain.

**Sliding window attention:** attend only to a fixed-size local window. Used in models designed for very long documents.

---

## ⚠️ Common confusions

**Confusion: attention scores directly measure semantic similarity.**
Attention scores measure relevance for the current task — learned during training. They are not purely semantic. A grammatically dependent word might get high attention even if semantically distant. Attention is task-driven, not purely semantic.

**Confusion: the attention matrix is fixed after training.**
The Wq, Wk, Wv matrices are fixed after training. But the attention scores (the n×n matrix) are computed fresh for every new input — they depend on the specific tokens in the sequence. The model's learned parameters are fixed; the attention patterns adapt to each input.

**Confusion: more attention heads always means better.**
More heads = more parameters = potentially better but also more compute and risk of overfitting on small datasets. GPT-3 uses 96 heads, but smaller models work well with 8–12. The right number depends on model size and task complexity.

**Confusion: self-attention and cross-attention are the same.**
Self-attention: Q, K, V all come from the same sequence — tokens attend to each other within one sequence. Cross-attention: Q comes from one sequence (decoder), K and V from another (encoder output) — used to connect encoder and decoder in seq2seq models.

---

## Interview-ready summary

> "Attention computes a new representation for each token by taking a weighted blend of all other tokens' Value vectors, where weights are determined by the dot product of the token's Query with every other token's Key, scaled by √dₖ and normalised by softmax. This creates direct connections between any two tokens regardless of distance — unlike RNNs which connect tokens sequentially. The O(n²) complexity means every token attends to every other, making it expensive for long sequences. Causal masking in decoders sets future token scores to -∞ before softmax, enforcing autoregressive generation. Multi-head attention runs h parallel attention operations each learning different relationship types. Cross-attention connects encoder and decoder by using decoder Queries against encoder Keys and Values. FlashAttention is the standard implementation — same mathematical result but cache-optimised for 2–4x speed improvement."

---

## Fire round questions (attempt before checking answers)

**Q1.** Walk through what happens mathematically when token "was" computes its new representation via self-attention in the sentence "The solo was epic." What does the attention weight for each token represent?

**Q2.** Why is the scaling factor √dₖ applied before softmax, and what goes wrong without it?

**Q3.** What is the difference between self-attention and cross-attention? Give a concrete example of where each is used.

**Q4.** Self-attention has O(n²) complexity. What does this mean practically for a model with a 128k context window, and how do efficient attention variants address this?

---

## Fire round answers

**Q1.** For "was" (position 3 in a decoder with causal masking), it can attend to "The", "solo", and "was" but not "epic" (future token — masked to -∞). The model computes Q₃·K₁, Q₃·K₂, Q₃·K₃, then divides by √dₖ, applies softmax. The result is three attention weights summing to 1.0. Each weight represents "how relevant is this past token to understanding was in this context?" The new representation of "was" is a weighted blend of V("The"), V("solo"), V("was") — now contextually enriched. Without masking, "was" would also see "epic" which would be information leakage — the model would know what comes next and couldn't be trained to predict it.

**Q2.** Dot products grow with the dimensionality dₖ. In a model with dₖ=512, the raw dot products can be in the range of ±100 or larger. When these large values go into softmax, the exponential function amplifies differences dramatically — one score dominates (close to 1.0) and all others approach 0. This creates extremely peaked attention distributions where the model effectively ignores most tokens. The gradients from softmax also vanish in these saturation regions — the model cannot learn to distribute attention. Dividing by √dₖ keeps scores in the ±2–5 range where softmax produces meaningful gradients and distributes attention across relevant tokens.

**Q3.** Self-attention: Q, K, V all come from the same sequence — every token attends to every other token within one sequence. Used in encoder layers (BERT attending to the full input) and decoder layers (GPT attending to past tokens). Cross-attention: Q comes from one sequence, K and V from another sequence. Used in encoder-decoder models — the decoder's Q represents "what am I trying to generate now?" and the encoder's K/V represent "what did the encoder understand about the input?" This allows the French decoder to directly attend to the relevant English words at each generation step in translation.

**Q4.** O(n²) means attention scores grow quadratically with sequence length. For 128k tokens: 128,000² = ~16 billion attention scores computed per layer. For a 96-layer model this is ~1.5 trillion operations per forward pass — prohibitively expensive. Efficient attention variants address this: FlashAttention keeps the same mathematical result but rewrites the computation to be cache-friendly (no full n×n matrix materialised in memory), achieving 2–4x speedup. Sparse attention (Longformer) reduces to O(n) by attending only to local windows plus global tokens, accepting a quality trade-off. Sliding window attention with some global tokens is a common practical choice for very long context models.

---

## Resources
- **Paper:** "Attention Is All You Need" — Vaswani et al. 2017
- **Paper:** "Neural Machine Translation by Jointly Learning to Align and Translate" — Bahdanau et al. 2014 (original attention paper)
- **YouTube:** 3Blue1Brown — "Attention in Transformers, Visually Explained"
- **YouTube:** Andrej Karpathy — "Let's build GPT from scratch"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
