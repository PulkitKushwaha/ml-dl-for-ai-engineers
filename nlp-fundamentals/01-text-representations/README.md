# Text Preprocessing & Representations

> **NLP Fundamentals · Topic 1** · How raw text becomes something a machine can process — from bag of words to transformer embeddings.

---

## The core idea

Every NLP technique in history was invented to fix a specific failure of the technique before it. Understanding the problem each technique solves — not just what it is — is what makes the evolution story click in interviews. The chain: One-hot → Bag of Words → TF-IDF → Word2Vec → Contextual Embeddings.

---

## The alien intelligence analogy

Think about teaching a completely alien intelligence to understand music reviews — an alien that only understands numbers.

Each attempt below is one step in the evolution. Each attempt fixes the previous failure and introduces a new one.

---

## Part 0 — Text Preprocessing Pipeline

Before any representation technique, raw text must be cleaned and normalised. This is the preprocessing pipeline.

### Tokenisation

Split text into units. Three levels:

**Word-level:** split on whitespace and punctuation.
```
"The guitar was epic." → ["The", "guitar", "was", "epic"]
```

**Sentence-level:** split on sentence boundaries.
```
"The guitar was epic. The solo was incredible." → 2 sentences
```
Harder than it looks — "Dr. Smith works at St. Mary's." has periods that are not sentence boundaries.

**Subword (BPE/WordPiece):** split into subword units — used by all modern LLMs. Covered in Topic 18 (Tokenisation & Embeddings).

### Lowercasing

"Guitar" → "guitar". Reduces vocabulary size — "Guitar" and "guitar" become the same token.

**Caution:** case sometimes carries meaning. "US" (country) vs "us" (pronoun). "Apple" (company) vs "apple" (fruit). Apply selectively — typically for BoW and TF-IDF but not for transformer models which can use case as a signal.

### Stop word removal

Remove high-frequency, low-information words: "the", "is", "at", "a", "and", "of".

**Why:** these words appear in almost every document and carry very little discriminative information. In a BoW or TF-IDF representation they add noise and increase dimensionality without adding signal.

**When NOT to remove stop words:** transformer models — they use stop words as grammatical signals. "Not good" vs "good" — removing "not" destroys meaning. Never apply stop word removal to transformer inputs.

### Stemming

Crude rule-based root extraction — chops word endings using rules.

```
"running"  → "run"
"guitars"  → "guitar"
"played"   → "play"
"better"   → "better"   ← wrong! should be "good"
"studies"  → "studi"    ← wrong! aggressive chopping
```

**Algorithm:** Porter Stemmer (most common). Applies a series of rules in sequence.

**When to use:** high-volume search indexes where speed matters more than linguistic accuracy. Elasticsearch uses stemming by default.

**Problem:** linguistically incorrect. "Better" → "better" (should map to "good"). "Studying" → "studi" (not a real word).

### Lemmatisation

Linguistically correct root using a dictionary lookup with part-of-speech context.

```
"running"  (verb) → "run"
"better"   (adj)  → "good"    ← correct!
"studies"  (verb) → "study"   ← correct!
"guitars"  (noun) → "guitar"
```

**Algorithm:** uses WordNet or similar lexical database. Needs POS tag to disambiguate.

**When to use:** when linguistic accuracy matters — information extraction, question answering, semantic analysis.

**The interview answer on stemming vs lemmatisation:**
> "Stemming is faster but crude — applies rules to chop word endings, sometimes incorrectly. Lemmatisation is slower but linguistically correct — uses a dictionary to find the actual base form. Use stemming for high-volume search indexes where speed matters. Use lemmatisation when linguistic accuracy is important."

### Punctuation handling

Remove punctuation for BoW/TF-IDF. Be careful with hyphenated terms: "state-of-the-art" → keep as one token or split? "C++" → removing + destroys meaning. Context-dependent.

### The full pipeline order

```
Raw text
  ↓ Sentence tokenisation (if needed)
  ↓ Word tokenisation
  ↓ Lowercasing
  ↓ Punctuation removal/normalisation
  ↓ Stop word removal
  ↓ Stemming OR Lemmatisation
  ↓ Clean token list → ready for representation
```

---

## Part 1 — One-Hot Encoding

### What it is

Vocabulary size V. Each word = V-dimensional vector with a 1 at its index, 0 everywhere else.

```
Vocabulary: {guitar=0, bass=1, solo=2, epic=3, bicycle=4, ...}   (V=50,000)

"guitar"  = [1, 0, 0, 0, 0, ...]
"bass"    = [0, 1, 0, 0, 0, ...]
"solo"    = [0, 0, 1, 0, 0, ...]
"bicycle" = [0, 0, 0, 0, 1, ...]
```

### Problems

**No semantic relationships:** dot product between any two different words = 0. "Guitar" and "bass" are as different as "guitar" and "bicycle." The alien has no idea these are related.

**Enormous dimensionality:** 50,000-dimensional sparse vectors. Most values are zero. Computationally wasteful.

**Still used for:** categorical features in classical ML (not NLP), label encoding.

---

## Part 2 — Bag of Words (BoW)

### What it is

Extend one-hot to documents. Document = vector of word counts. Vocabulary-sized vector where each position = count of that word in the document.

```
Doc: "The guitar was epic. The solo was incredible."

After preprocessing: ["guitar", "epic", "solo", "incredible"] (stop words removed)

Vocabulary: {guitar:0, epic:1, solo:2, incredible:3, bass:4, ...}

BoW vector: [1, 1, 1, 1, 0, ...] ← count of each vocab word
```

**Document similarity:** two documents with similar BoW vectors have similar word distributions → likely about similar topics. This is the intuition behind search: find documents with high overlap with the query BoW.

### Problems

**Common words dominate:** if we don't remove stop words, "the" appears 2 times and dominates. Even after stop word removal, moderately common words ("good", "great", "said") dilute signal.

**No word order:** "Dog bites man" = "Man bites dog" — same BoW vector. The "bag" metaphor is literal — all words thrown in a bag, order destroyed.

**No word importance:** all words weighted equally. "Guitar" appearing once has the same weight as "the" appearing once — even though "guitar" is far more informative.

**Still used for:** simple text classification baselines, spam detection, topic modelling (LDA), document similarity in resource-constrained systems.

---

## Part 3 — TF-IDF

### What it solves

TF-IDF fixes the word importance problem — rare, distinctive words should get higher weight than common words.

### The formula

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

**Term Frequency TF(t, d):** how often term t appears in document d, normalised by document length.
```
TF(t, d) = count(t in d) / total_words_in_d
```

**Inverse Document Frequency IDF(t):** log of how rare term t is across all documents.
```
IDF(t) = log( N / (1 + df(t)) )
```
Where N = total documents, df(t) = documents containing t. The +1 prevents division by zero.

**Why log?** Without log, IDF grows too fast for very rare terms. Log compresses the scale — a term in 1 document vs 10 documents is more important than the same difference at 10,000 vs 10,001 documents.

### Concrete example

```
Corpus: 10,000 music reviews

"the"       → appears in 10,000/10,000 docs → IDF = log(10000/10000) = 0 → weight = 0
"guitar"    → appears in 3,000/10,000 docs  → IDF = log(10000/3000) = 1.2
"solo"      → appears in 1,000/10,000 docs  → IDF = log(10000/1000) = 2.3
"phrygian"  → appears in 3/10,000 docs      → IDF = log(10000/3) = 8.1

Document: "The phrygian solo was breathtaking"
TF-IDF scores: the=0, phrygian=0.81, solo=0.23, was=~0, breathtaking=high
```

"Phrygian" gets very high weight — it's rare and therefore highly informative. "The" gets zero weight — it's in every document and tells you nothing discriminative.

### TF-IDF in search

Query = treated as a short document. Score each corpus document by TF-IDF similarity to query. Return highest-scoring documents. This is the foundation of traditional search — Google used TF-IDF variants until the neural era.

### Remaining problems

**No word order:** "guitar plays solo" = "solo plays guitar" — same TF-IDF vector.

**No semantic relationships:** "guitar" and "instrument" are completely different dimensions. A document about "stringed instruments" will not match a query for "guitar techniques."

**Vocabulary mismatch:** user asks "how do I fix buzzing on my instrument" — no documents contain all these words. A highly relevant document "guitar fret buzz repair guide" scores low because the vocabulary doesn't overlap.

**Sparse and high-dimensional:** still one dimension per vocabulary term.

**Still used for:** traditional search engines, BM25 (improved TF-IDF), document clustering, keyword extraction, as the sparse component in hybrid RAG.

---

## Part 4 — Word2Vec & GloVe (Static Embeddings)

### What it solves

Word2Vec encodes semantic relationships — "guitar" and "bass" become nearby in vector space because they appear in similar contexts.

### How Word2Vec works

Train a shallow neural network on a massive text corpus with one of two objectives:

**CBOW (Continuous Bag of Words):** given surrounding context words, predict the centre word.
```
Context: ["The", "was", "epic"] → predict: "solo"
Context: ["The", "was", "epic"] → predict: "guitar"
```

**Skip-gram:** given a centre word, predict surrounding context words.
```
"solo" → predict: ["The", "guitar", "was", "epic"]
```

The key insight: words that appear in similar contexts will develop similar weight vectors in the neural network — because the network learns to predict the same context words for them. After training, throw away the prediction layer. The learned weight vectors are the word embeddings.

### Why semantic relationships emerge

"Guitar" and "bass" both frequently appear near "strings", "play", "music", "fret", "chord". The neural network learns to predict these same context words for both. Therefore their weight vectors become similar.

"King" and "queen" both appear near "royal", "throne", "crown", "palace", "reign". Their vectors become similar. But "king" appears near "man", "prince", "his". "Queen" appears near "woman", "princess", "her". So: queen ≈ king − man + woman. The arithmetic relationship emerges from training context statistics.

### GloVe (Global Vectors)

Similar goal to Word2Vec but different training approach. Instead of local context windows, GloVe uses global co-occurrence statistics — how often does word i appear near word j across the entire corpus? Factorises the co-occurrence matrix to produce embeddings.

**In practice:** GloVe and Word2Vec produce similar quality embeddings. GloVe trains faster. Word2Vec scales better to very large corpora.

### The famous arithmetic

```
king − man + woman ≈ queen
Paris − France + Germany ≈ Berlin
guitar − music + sport ≈ tennis (roughly)
```

Nobody programmed these relationships. They emerge from the statistics of language — from which words appear near which other words across billions of sentences.

### The remaining problem — polysemy

One fixed vector per word regardless of context.

```
"I went to the bank to deposit money"     → bank = [same vector]
"I sat by the river bank watching fish"   → bank = [same vector]
"The bank of monitors lit up the room"    → bank = [same vector]
```

All three sentences — completely different meanings of "bank" — produce exactly the same embedding for "bank." The vector is usually biased toward the most common meaning in the training corpus (financial institution).

**This is exactly what Transformers fix.**

---

## Part 5 — Contextual Embeddings (BERT, GPT, Transformers)

### What it solves

Every word gets a different vector depending on its full context. Same word, completely different representations in different sentences.

### How it works

The Transformer's attention mechanism computes each token's representation by attending to all other tokens in the sequence. The resulting vector for any token is a blend of all other tokens' information — weighted by how relevant they are.

```
"I went to the bank to deposit money"
bank → vector blended with: went, deposit, money → finance-domain representation

"I sat by the river bank watching fish"
bank → vector blended with: river, watching, fish → nature-domain representation
```

The word "bank" produces a completely different vector in each sentence because the attention weights are completely different — "deposit" and "money" pull the first "bank" toward finance; "river" and "fish" pull the second toward nature.

### What this enables

**Polysemy:** same word, different meanings, different vectors. Solved.

**Long-range dependencies:** "The guitarist who toured with Miles Davis in 1967 was ___." Attention directly connects "guitarist" (word 2) to the blank (word 12) — no compression through sequential hidden states.

**Rich contextual representations:** every token's vector encodes not just what the word means in isolation but what it means in this specific context, in this specific sentence, for this specific task.

### Two types of contextual embeddings in practice

**Token-level embeddings:** the vector for each individual token in context. Used when you need word-level analysis.

**Sentence/document embeddings:** a single vector representing the whole input. Usually produced by taking the [CLS] token (BERT) or averaging all tokens (sentence transformers). Used in RAG retrieval — embed query and all document chunks, find nearest neighbours.

---

## The complete evolution — why each step was needed

```
One-hot encoding
→ Problem: no relationships, enormous sparse vectors
→ Fix: Bag of Words (represents documents as word counts)

Bag of Words
→ Problem: common words dominate, no word importance
→ Fix: TF-IDF (weights words by rarity/importance)

TF-IDF
→ Problem: vocabulary mismatch, no semantic relationships, sparse
→ Fix: Word2Vec/GloVe (dense semantic vectors via neural training)

Word2Vec/GloVe
→ Problem: one fixed vector per word, cannot handle polysemy
→ Fix: Transformer embeddings (context-dependent vectors via attention)

Transformer embeddings
→ Current standard — handles polysemy, long-range dependencies, rich context
```

---

## Full comparison

| Property | One-hot | BoW | TF-IDF | Word2Vec/GloVe | BERT/GPT |
|---|---|---|---|---|---|
| **Dimensions** | V (50k+) | V (50k+) | V (50k+) | 300 | 768–4096 |
| **Dense?** | No | No | No | Yes | Yes |
| **Semantic relationships** | None | None | None | Static | Contextual |
| **Polysemy** | No | No | No | No | Yes |
| **Word order** | No | No | No | Partial (window) | Full (attention) |
| **Training required** | No | No | No | Shallow NN | Large Transformer |
| **Still used?** | Rarely | Basic classification | Search/BM25 | Legacy systems | Yes — standard |

---

## ⚠️ Common confusions

**Confusion: TF-IDF and BoW are the same thing.**
BoW = raw word counts. TF-IDF = word counts weighted by corpus-wide rarity. TF-IDF is BoW plus the IDF weighting step. The key difference: TF-IDF gives "the" a weight of ~0 and "phrygian" a very high weight. BoW treats both identically based on raw count.

**Confusion: Word2Vec understands context like BERT.**
Word2Vec uses a fixed context window (typically 5 words) during training, but the resulting vectors are static — one fixed vector per word regardless of usage context. BERT computes fresh contextual vectors at inference time using full-sequence attention. Word2Vec captures that similar words appear in similar contexts during training. BERT captures what each specific word means in each specific usage.

**Confusion: stemming and lemmatisation produce the same results.**
Stemming applies rules to chop endings — fast but linguistically incorrect. Lemmatisation uses a dictionary with POS context — slower but correct. "Better" → stemming gives "better", lemmatisation gives "good." "Studies" → stemming gives "studi", lemmatisation gives "study." For search indexes, stemming is usually sufficient. For language understanding tasks, lemmatisation is preferred.

**Confusion: stop word removal is always a good idea.**
For BoW and TF-IDF: yes — stop words add noise and inflate dimensionality without adding signal. For transformer models: no — stop words carry grammatical information that changes meaning. "Not good" vs "good" — removing "not" completely changes the meaning. Never apply stop word removal before feeding text to a transformer.

**Confusion: contextual embeddings made TF-IDF obsolete.**
TF-IDF (and its improvement BM25) is still widely used in production search systems because it is fast, interpretable, requires no training, and excels at exact keyword matching. Modern production search (Elasticsearch, hybrid RAG) combines BM25 sparse retrieval with dense embedding retrieval — not replacing one with the other.

---

## Interview-ready summary

> "Text representations evolved as a problem-solution chain. One-hot encoding has no relationships between words. Bag of Words represents documents as word count vectors but common words dominate. TF-IDF weights words by rarity across the corpus — rare discriminative words get high weight, common words get near-zero weight — but still has no semantic relationships. Word2Vec trains a shallow neural network to predict words from context, producing dense 300-dimensional vectors where semantic relationships emerge: guitar and bass become nearby, king − man + woman ≈ queen. But Word2Vec gives one fixed vector per word regardless of context — bank means the same thing in financial and river-bank sentences. Transformer contextual embeddings (BERT, GPT) fix this: every token's vector is computed fresh using attention over the full input sequence — same word produces completely different vectors in different contexts. The preprocessing pipeline (tokenisation, lowercasing, stop word removal, stemming/lemmatisation) is applied before BoW/TF-IDF but NOT before transformers."

---

## Resources
- **Library:** `sklearn.feature_extraction.text` — TfidfVectorizer, CountVectorizer
- **Library:** `gensim` — Word2Vec, GloVe training
- **Library:** `nltk` — stop words, stemming (PorterStemmer), lemmatisation (WordNetLemmatizer)
- **Library:** `spacy` — lemmatisation, POS tagging, NER
- **Paper:** "Efficient Estimation of Word Representations in Vector Space" — Mikolov et al. 2013 (Word2Vec)
- **Paper:** "GloVe: Global Vectors for Word Representation" — Pennington et al. 2014

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
