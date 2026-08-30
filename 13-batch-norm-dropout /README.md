# 13 — Batch Normalisation & Dropout

> **Week 2 · Topic 13** · Two techniques that transformed deep learning — one stabilises training, one prevents overfitting.

---

## The core idea

Batch normalisation and dropout solve two different problems in neural networks:

- **Batch normalisation** — fixes internal covariate shift, stabilises training, allows higher learning rates, dramatically speeds up convergence
- **Dropout** — prevents overfitting by randomly deactivating neurons during training, forcing redundant representations

Both are regularisation techniques in the broad sense — both help the network generalise rather than memorise. But they solve different root causes and are often used together.

---

## Part 1 — Batch Normalisation

### The problem — internal covariate shift

As training progresses, weights in early layers change. This changes the distribution of their outputs — which are the inputs to the next layer. That layer was adapting to one distribution, now it faces a different one. Then its weights change, shifting the distribution for the layer after it.

Every layer is constantly learning from a moving target. This is **internal covariate shift** — it slows training significantly, requires very careful learning rate tuning, and makes deep networks unstable.

### The recording studio analogy

Your band records a long session. In the morning the drummer hits hard — levels are loud. After lunch everyone is tired — levels drop. By evening with coffee — levels spike. Your sound engineer constantly chases these shifting levels, adjusting everything downstream every time.

Batch normalisation is a sound engineer who automatically normalises levels between every recording session — before they reach the next stage. Each layer always receives input with mean≈0 and std≈1. No more chasing shifting distributions. Everyone focuses on learning.

### How batch normalisation works — 3 steps

**Step 1 — Normalise the batch**

For each mini-batch, compute the mean and variance of activations at this layer. Subtract the mean and divide by the standard deviation:

```
x̂ᵢ = (xᵢ − μ_batch) / √(σ²_batch + ε)
```

Result: activations have mean≈0, std≈1 across the batch.

**Step 2 — Scale (γ)**

Multiply normalised activations by a learned parameter γ:

```
yᵢ = γ · x̂ᵢ
```

**Step 3 — Shift (β)**

Add a learned parameter β:

```
yᵢ = γ · x̂ᵢ + β
```

### Why γ and β? Why not just normalise?

Pure normalisation to mean=0, std=1 might be too restrictive — some layers may genuinely need a different scale or mean to represent their features. γ and β are learned during training, giving the network freedom to undo the normalisation if that is optimal. In practice they often converge to γ≈1 and β≈0, confirming the normalisation was helpful. But the network has the choice.

### Placement in the network

Standard pattern at each hidden layer:

```
Linear (weighted sum) → BatchNorm → Activation (ReLU) → Dropout
```

- Apply to hidden layers — not the input layer (that is feature scaling, done separately)
- Never apply to the output layer
- γ and β are learned parameters — they get gradients and update during backprop

### Train vs inference — a critical difference

**During training:**
Batch norm uses the mean and variance of the current mini-batch to normalise. Different batches give slightly different normalisations — this adds a small amount of noise that acts as regularisation. Running averages of mean and variance are accumulated:

```
running_mean = 0.9 × running_mean + 0.1 × batch_mean
running_var  = 0.9 × running_var  + 0.1 × batch_var
```

**During inference:**
No mini-batch available — you might predict one sample at a time. Batch norm uses the running averages accumulated during training. Predictions are deterministic and consistent.

**The interview trap:** does batch norm behave the same at training and inference? **No.** Forgetting to switch modes causes subtle bugs — predictions at inference are stochastic instead of deterministic.

```python
model.train()   # training mode — uses batch statistics
model.eval()    # inference mode — uses running averages
```

Always call `model.eval()` before inference in PyTorch.

### Benefits of batch normalisation

| Benefit | Why |
|---|---|
| **Faster training** | Stable activations allow higher learning rates — often 2–10x speedup |
| **Reduces vanishing gradient** | Keeps activations in healthy range — gradients flow consistently through deep networks |
| **Acts as regularisation** | Batch-level noise reduces overfitting — sometimes reduces need for dropout |
| **Less sensitive to initialisation** | Bad weight initialisation is corrected at each layer |

---

## Part 2 — Dropout

### The problem — co-adaptation of neurons

Without dropout, neurons can co-adapt. Neuron A learns to always correct for Neuron B's mistakes. They become dependent on each other. When new data arrives with slightly different patterns, this co-adaptation breaks down — overfitting.

### The band rehearsal analogy

Your band has 5 members. If all 5 always rehearse together, they cover for each other's weaknesses — but only as a unit. Randomly, each rehearsal, one or two members call in sick. Nobody knows who won't show. Every member must learn to play well regardless of who else is there. The guitarist learns to hold the low end. The drummer carries the rhythm without the bassist. Each member becomes more capable independently.

That is dropout. Each training step, random neurons switch off. Every neuron learns to be useful regardless of which others are active. The result: a more robust, generalised network.

### How dropout works

During each training step, every neuron is independently switched off with probability p (the dropout rate):

```
Training step:
  Neuron 47:  z=1.12  → DROP  → output = 0.00
  Neuron 301: z=0.67  → keep  → output = 0.67 × (1/1-p) = 1.34
  Neuron 412: z=0.89  → DROP  → output = 0.00
```

Surviving activations are scaled by 1/(1−p) to maintain expected magnitude.

**During backpropagation:** the same dropout mask is reused. Dropped neurons receive zero gradient — their weights are not updated. Consistent — if a neuron did not contribute to the prediction, it is not penalised or rewarded.

**During inference:** dropout is switched off — all neurons active. No scaling needed because training already used inverted dropout scaling.

### Why dropout prevents overfitting

Dropout prevents co-adaptation — neurons cannot rely on specific other neurons always being present. Each neuron must learn to be useful independently.

Another view: dropout trains an exponential number of different subnetworks simultaneously (2ⁿ subnetworks for n neurons) and averages their predictions at inference — similar to how random forests average many trees. Ensemble learning built into one network.

### Dropout rate guidelines

| Rate | Effect | Use when |
|---|---|---|
| p=0.1–0.2 | Light regularisation | Input layer, convolutional layers, Transformers |
| p=0.3–0.5 | Standard regularisation | Hidden layers in dense networks |
| p>0.6 | Heavy regularisation | Rarely used — risks underfitting |

```
Practical guidelines:
- Hidden layers:  p=0.2–0.5 (start with 0.5, reduce if underfitting)
- Input layer:    p=0.1–0.2 (lower — don't lose too much input signal)
- Output layer:   NEVER apply dropout
- Transformers:   typically p=0.1
```

---

## Where regularisation fits in the full training lifecycle

Regularisation appears in three places during neural network training:

### 1. Forward pass — Dropout

Every iteration, a random dropout mask is applied at each hidden layer. This is where co-adaptation is prevented. The mask changes every iteration — a completely different set of neurons is dropped each time.

### 2. Forward pass — Batch Normalisation noise

Batch statistics (mean and variance) computed on each mini-batch are slightly noisy — they vary from batch to batch. This noise prevents the network from memorising specific training examples. A weak but consistent regularisation effect throughout training.

### 3. Weight update step — L2 weight decay

Classical L2 regularisation plugs directly into the weight update:

```
Without regularisation:
w = w − learning_rate × gradient

With L2 weight decay:
w = w − learning_rate × (gradient + λ × w)
```

The extra `λ × w` term continuously shrinks all weights toward zero — preventing any single weight from growing large and dominating predictions. In PyTorch:

```python
optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.001,
                             weight_decay=0.0001)  # λ = 0.0001
```

### 4. Training loop — Early stopping

Monitor validation loss after each epoch. Stop training when validation loss stops improving for a set number of epochs (patience). Prevents overfitting after the optimal epoch — even if training loss is still falling.

### Complete regularisation picture

| Where in lifecycle | Technique | Prevents |
|---|---|---|
| Weight update step | L2 weight decay | Weights growing too large |
| Hidden layers 1–2 forward pass | Dropout p=0.5 | Co-adaptation, memorisation |
| Hidden layers 3–4 forward pass | Dropout p=0.3 | Same, lighter touch |
| Every hidden layer forward pass | Batch norm noise | Weak memorisation |
| Training loop | Early stopping | Overfitting past optimal epoch |

---

## Batch normalisation vs Dropout — comparison

| Property | Batch Normalisation | Dropout |
|---|---|---|
| **Primary purpose** | Training stability | Overfitting prevention |
| **Solves** | Internal covariate shift | Co-adaptation of neurons |
| **Mechanism** | Normalises layer inputs per batch | Randomly deactivates neurons |
| **Speeds training?** | Yes — significantly | No — slightly slows |
| **Regularisation?** | Yes — side effect | Yes — primary purpose |
| **Inference behaviour** | Uses running averages — different from training | Switched off — all neurons active |
| **Used together?** | Yes — commonly in CNNs and dense networks | Yes |

**Note on Transformers:** Layer Normalisation is preferred over Batch Normalisation in Transformer architectures — it normalises across features for each sample rather than across the batch. Covered in the Transformer topic.

---

## ⚠️ Common confusions

**Confusion: batch norm behaves the same at training and inference.**
No. During training it uses batch statistics (stochastic). During inference it uses running averages (deterministic). Always call `model.eval()` before inference. Forgetting this causes predictions to vary between runs — a subtle production bug.

**Mistake: applying dropout to the output layer.**
Never apply dropout to the output layer. The final prediction needs all neurons active and stable — randomly zeroing output neurons destroys the prediction entirely.

**Confusion: batch norm removes the need for dropout.**
Batch norm's regularisation effect is a side effect of batch-level noise — not a substitute for dropout. On large networks with many parameters, overfitting can still occur with batch norm alone. Both are often used together.

**Confusion: dropout is active during inference.**
Dropout is only active during training. During inference all neurons are active. The 1/(1−p) scaling applied during training (inverted dropout) ensures the expected activation magnitude is correct at inference without any additional adjustment.

**Mistake: not scaling surviving activations by 1/(1−p) during training.**
Without scaling, the expected activation magnitude at inference (where all neurons are active) would be higher than during training (where only 1−p fraction were active). This causes a distribution mismatch. Inverted dropout (scaling during training) is the standard solution.

---

## Interview-ready summary

> "Batch normalisation normalises the inputs to each hidden layer per mini-batch — subtracting the batch mean and dividing by batch standard deviation, then applying learned scale (γ) and shift (β) parameters. This fixes internal covariate shift, stabilises gradient flow, allows higher learning rates, and speeds training. Critically, it behaves differently at inference — using running averages instead of batch statistics, so model.eval() must be called before inference. Dropout randomly deactivates neurons with probability p during training — forcing neurons to learn independent representations and preventing co-adaptation. At inference dropout is switched off. Regularisation in a full neural network comes from multiple sources working together: L2 weight decay in the weight update step, dropout in the forward pass, batch norm noise as a side effect, and early stopping at the training loop level."

---

## Resources
- **Udemy:** Deep Learning A-Z — Kirill Eremenko (Part 1: ANNs — dropout and regularisation sections)
- **YouTube:** StatQuest — "Batch Normalisation, Clearly Explained"
- **YouTube:** StatQuest — "Dropout, Clearly Explained"

---

*Part of [ml-dl-for-ai-engineers](https://github.com/PulkitKushwaha/ml-dl-for-ai-engineers) — a learning journal built while targeting Agentic AI Engineer roles at product companies.*
