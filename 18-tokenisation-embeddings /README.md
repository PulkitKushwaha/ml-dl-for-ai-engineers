# 18 — Tokenisation & Embeddings

> **Week 2 · Topic 18** · How text becomes numbers — the bridge between human language and neural network computation.

---

## The core idea

Neural networks cannot process raw text — they only process numbers. Tokenisation splits text into manageable units (tokens), and embeddings convert those tokens into dense vectors that encode semantic meaning. These two steps are the entry point of every LLM pipeline and directly affect agent performance, context window usage, and retrieval quality.

---

## Why this topic matters for your role

As an Agentic AI Engineer you work with tokenisation and embeddings every day:
- Context window limits are measured in tokens — understanding tokenisation tells you why "Hello world" ≠ 2 tokens
- RAG pipelines depend on embedding quality — the retrieval component is entirely an embedding problem
- Prompt engineering costs money per token — knowing how tokenisation works helps optimise prompt efficiency
- Chunking strategies for RAG depend on understanding how embeddings capture semantic meaning

---

## Part 1 — Tokenisation

### What is a token?

A token is the basic unit of text that an LLM processes. Tokens are not always words — they can be subwords, characters, or word fragments depending on the tokenisation algorithm.

```
"Hello world"          → ["Hello", " world"]              = 2 tokens
"Transformer"          → ["Trans", "former"]              = 2 tokens
"unbelievable"         → ["un", "believ", "able"]         = 3 tokens
"GPT-4"               → ["G", "PT", "-", "4"]            = 4 tokens
"   "  (3 spaces)     → ["   "]                          = 1 token
"\n\n"                → ["\n\n"]                         = 1 token
```

### Tokenisation algorithms

**Character-level tokenisation:**
Split every character into its own token. Simple but creates very long sequences — "hello" = 5 tokens. Rare characters handled naturally. Largely obsolete for LLMs.

**Word-level tokenisation:**
Split on whitespace and punctuation. "Hello world" = ["Hello", "world"]. Problem: vocabulary explosion — millions of unique words, rare words become unknown tokens (OOV — out of vocabulary). Also poor at handling morphology ("run", "running", "runner" are completely separate tokens with no shared representation).

**Byte-Pair Encoding (BPE) — GPT family:**
Start with individual characters. Iteratively merge the most frequent adjacent pairs into a single token. Repeat until vocabulary size is reached (GPT-4 uses ~100,000 tokens).

```
Training data corpus → count all adjacent pairs
Most frequent: "e" + "r" → merge into "er"
Next most frequent: "er" + "s" → merge into "ers"
...continues until vocabulary size reached
```

Result: common words become single tokens, rare words get split into subword units. "playing" → single token. "antidisestablishmentarianism" → multiple subword tokens. Never produces OOV — any character sequence can be encoded.

**WordPiece — BERT family:**
Similar to BPE but merges based on likelihood rather than frequency. Subword pieces marked with "##" to indicate continuation: "playing" → ["play", "##ing"]. Handles morphology better than word-level.

**SentencePiece — LLaMA, T5:**
Language-agnostic BPE/unigram tokenisation that treats text as a raw byte stream — handles any language, special characters, and whitespace uniformly.

### Why tokenisation choices matter in practice

**Context window:** GPT-4's 128k context window = 128,000 tokens. A typical English word ≈ 1.3 tokens. So 128k tokens ≈ ~100,000 words ≈ a short novel.

**Multilingual efficiency:** English is typically well-compressed (~1.3 tokens/word). Other languages can be much less efficient — some languages take 3–5 tokens per word because the training data was English-dominated and rare characters get split aggressively.

**Prompt costs:** API pricing is per token. "Summarise this document:" = 4 tokens. Understanding tokenisation helps write efficient prompts and estimate costs accurately.

**Code tokenisation:** code is typically more token-efficient than English for common syntax (`def`, `for`, `if` are single tokens) but less efficient for variable names.

### The tokenisation → integer → embedding pipeline

```
Raw text  →  Tokeniser  →  Token IDs  →  Embedding layer  →  Dense vectors
"Hello"   →  "Hello"   →  15496      →  [0.2, -0.8, ...]  →  512-dim vector
```

Each token in the vocabulary has an integer ID. The embedding layer is a lookup table — given an integer ID, return its learned embedding vector.

---

## Part 2 — Embeddings

### What is an embedding?

An embedding is a dense vector (typically 128–4096 dimensions) that represents a token, sentence, or document in a continuous mathematical space where **geometric proximity = semantic similarity**.

```
Semantic similarity in embedding space:
  "guitar"  ↔  "bass"     → small distance (both instruments)
  "guitar"  ↔  "piano"    → medium distance (both musical but different family)
  "guitar"  ↔  "bicycle"  → large distance (unrelated)
```

### Why dense vectors instead of one-hot encoding?

**One-hot encoding:** vocabulary of 50,000 words → each word is a 50,000-dimensional vector with a single 1 and all zeros. Problems: (1) no semantic relationships encoded — "king" and "queen" are as different as "king" and "bicycle," (2) extremely high-dimensional and sparse, (3) no way to handle unseen words.

**Dense embeddings:** 512-dimensional vector with continuous values. Benefits: (1) semantic relationships encoded via proximity, (2) compact — 512 dimensions vs 50,000, (3) arithmetic relationships emerge: king − man + woman ≈ queen.

### Types of embeddings

**Static embeddings (Word2Vec, GloVe):**
Each word has one fixed embedding regardless of context. "Bank" has the same vector whether it means a financial institution or a river bank. Trained on large corpora using shallow neural networks. Fast to compute but cannot capture polysemy (multiple meanings).

**Contextual embeddings (from Transformers):**
Every token's embedding is computed fresh based on its full context. "Bank" in "river bank" produces a completely different vector than "bank" in "bank account" — the Transformer's attention mechanism produces context-dependent representations. This is what makes modern LLMs powerful.

**Sentence/document embeddings:**
A single vector representing an entire sentence or document. Usually produced by an encoder model (BERT family) — often by taking the [CLS] token representation or averaging all token representations. Used in RAG retrieval — embed query and documents, find nearest neighbours.

### The embedding space — geometric properties

**Cosine similarity:** measures the angle between two vectors (not distance). Values from -1 (opposite) to +1 (identical). Standard metric for semantic similarity:
```
similarity("guitar", "bass") = 0.82   (high — related)
similarity("guitar", "bicycle") = 0.11  (low — unrelated)
```

**Arithmetic in embedding space:**
```
king − man + woman ≈ queen
Paris − France + Italy ≈ Rome
```
These arithmetic relationships emerge from training — nobody programmed them. The model learned that "capital city" is a consistent direction in embedding space.

**Clustering:** similar concepts cluster together. All musical instruments cluster near each other, all sports cluster near each other. This is what makes semantic search possible — search for "string instrument" and find vectors near guitar, bass, violin, cello.

### Embeddings in the Transformer

Inside a Transformer, there are multiple types of embeddings:

**Token embeddings:** the lookup table. Input token IDs → dense vectors. Learned during training. Shape: (vocab_size × d_model). For GPT-3: (50,257 × 12,288).

**Positional embeddings:** added to token embeddings to encode position. Either learned (GPT) or fixed sinusoidal functions (original Transformer). Same shape as token embeddings.

**The combined input:**
```
input_representation = token_embedding + positional_embedding
```

This combined vector is what flows into the first attention layer.

### Embedding models for RAG

In RAG pipelines, the retrieval component uses dedicated embedding models:

| Model | Dimensions | Use case |
|---|---|---|
| text-embedding-ada-002 (OpenAI) | 1,536 | General purpose, good quality |
| text-embedding-3-large (OpenAI) | 3,072 | Higher quality, more expensive |
| BGE-large (open source) | 1,024 | Strong open-source alternative |
| E5-large (open source) | 1,024 | Strong for retrieval tasks |

**Chunking strategy matters:** documents are split into chunks before embedding. Too large — the embedding loses specificity, retrieval is noisy. Too small — loses context, fragments meaning. Typical: 256–512 tokens per chunk with 50-token overlap between chunks.

---

## Tokenisation vs Embedding — the distinction

```
Tokenisation:   text → integer IDs   (discrete, lossless, deterministic)
                "Hello" → 15496

Embedding:      integer ID → dense vector   (continuous, learned, semantic)
                15496 → [0.2, -0.8, 0.5, ...]

They are sequential steps — tokenisation always comes before embedding.
Tokenisation is a preprocessing step. Embedding is the first learned layer.
```

---

## ⚠️ Common confusions

**Confusion: tokens = words.**
Tokens are subword units — not necessarily words. Common words are single tokens, rare or long words are split into multiple tokens. "playing" = 1 token. "antidisestablishmentarianism" = multiple tokens. Token count ≠ word count.

**Confusion: embeddings are fixed like a dictionary.**
Static embeddings (Word2Vec) are fixed per word. But Transformer contextual embeddings are computed fresh per input — the same word produces a different embedding depending on its context. "Bank" in two different sentences produces two completely different vectors.

**Confusion: larger embedding dimensions are always better.**
Larger dimensions capture more information but are slower to compute, more expensive to store, and require more training data to populate meaningfully. The right dimension depends on the task — 384 dimensions is often sufficient for retrieval, while 4,096 may be needed for complex reasoning tasks.

**Confusion: cosine similarity and Euclidean distance are interchangeable.**
Cosine similarity measures angle — direction of the vector, ignoring magnitude. Euclidean distance measures absolute distance. For embeddings, cosine similarity is preferred because it is invariant to the magnitude of the vector — two embeddings can be very close directionally even if one has a much larger norm.

**Confusion: the embedding layer is separate from the Transformer.**
The token embedding lookup table and the positional encoding are the first components of the Transformer — not a separate preprocessing step. They are learned jointly with the rest of the model during training.

---

## Interview-ready summary

> "Tokenisation splits text into subword units using algorithms like BPE (GPT) or WordPiece (BERT) — common words become single tokens, rare words get split, and no out-of-vocabulary problem exists. Each token maps to an integer ID. Embeddings convert these IDs to dense vectors in a continuous semantic space where geometric proximity equals semantic similarity. Static embeddings (Word2Vec) give each word one fixed vector. Contextual embeddings from Transformers compute each token's representation fresh based on full context via attention — the same word gets different vectors in different sentences. In RAG pipelines, encoder models produce sentence/document embeddings used for similarity search. Token count matters for context window limits, API costs, and chunking strategy — English averages ~1.3 tokens per word."

---

## Fire round questions (attempt before checking answers)

**Q1.** Why does BPE tokenisation never produce an out-of-vocabulary token, and why does this matter for LLMs?

**Q2.** Your RAG pipeline retrieves irrelevant documents even though the query seems clear. What embedding-related issues might be causing this and how would you fix them?

**Q3.** What is the difference between a static embedding and a contextual embedding? Give a concrete example where they would produce different results.

**Q4.** A colleague says "tokens and words are the same thing — a 10-word sentence is 10 tokens." Why is this wrong and what are the practical implications?

---

## Fire round answers

**Q1.** BPE starts with individual characters and iteratively merges frequent pairs until the vocabulary is built. Since the base vocabulary includes all individual bytes/characters, any string — no matter how unusual — can be encoded by falling back to character-level tokens. This matters because LLMs trained on English would fail completely on new words, code, or other languages if they used word-level tokenisation and encountered OOV tokens. BPE ensures any input can always be encoded.

**Q2.** Several embedding issues could cause poor retrieval: (1) Chunk size mismatch — chunks too large make embeddings too diffuse, losing specificity; reduce chunk size to 256–512 tokens. (2) Wrong embedding model — a general-purpose model may not capture domain-specific semantics; switch to a domain-specific or better-quality model. (3) Query-document asymmetry — the query is short and the documents are long, producing embeddings in different regions of the space; use asymmetric embedding models designed for retrieval (e.g. E5, BGE). (4) Missing overlap between chunks — context at chunk boundaries is lost; add 50-token overlap between chunks.

**Q3.** Static embeddings (Word2Vec) assign one fixed vector per word regardless of context. Contextual embeddings (Transformer) compute fresh vectors per token based on full surrounding context. Concrete example: "I went to the bank to deposit money" vs "I sat on the river bank." In Word2Vec, "bank" gets the same vector in both sentences — probably biased toward the financial meaning since it's more common in training data. In a Transformer, "bank" in the first sentence produces a vector close to "finance/money" while "bank" in the second produces a vector close to "river/nature." The contextual embedding captures polysemy — multiple meanings of the same word — which static embeddings cannot.

**Q4.** Tokens are subword units, not words. BPE splits rare and long words into multiple tokens while keeping common words as single tokens. Practical implications: (1) Context window limits — a "10,000-word document" might be 13,000–15,000 tokens, potentially exceeding the context window. (2) API costs — billed per token, not per word; multilingual text costs more because non-English languages tokenise less efficiently. (3) Prompt engineering — counting words to estimate token usage is inaccurate; use a tokeniser to get exact counts. (4) Chunking — splitting documents by word count for RAG gives inconsistent chunk token sizes; always chunk by token count.

---

## Resources
- **Udemy:** LLM Engineering — Ed Donner (tokenisation and embeddings sections)
- **Tool:** platform.openai.com/tokenizer — visualise how GPT tokenises any text
- **YouTube:** Andrej Karpathy — "Let's build the GPT Tokenizer"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
