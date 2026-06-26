# Architecture Concepts Reference

This file is the technical ground truth the analysis depends on. Read it before
writing Step 2 analysis so explanations are correct, consistent, and use the
right vocabulary. It distills the concepts Sebastian Raschka covers in "The Big
LLM Architecture Comparison", "Recent Developments in LLM Architectures", "A
Visual Guide to Attention Variants", and the DeepSeek technical tour, plus the
original technical reports.

## Table of contents
1. The shared decoder skeleton
2. Normalization (LayerNorm → RMSNorm, placement, QK-Norm, depth-scaling)
3. Feed-forward (SwiGLU)
4. Positional encoding (RoPE, ABF, YaRN, DCA, NoPE)
5. Attention family (MHA → MQA → GQA → MLA)
6. Long-context attention (sliding window, sparse/DSA, lightning/linear, CCA)
7. KV-cache math and sharing
8. Mixture of Experts (routing, granularity, shared experts, load balancing)
9. Multi-Token Prediction (MTP)
10. Efficiency: FP8, quantization, per-layer embeddings, gated attention
11. Post-training (RLVR, GRPO, CISPO, self-verification, distillation)
12. The analysis lens — what to actually compare

---

## 1. The shared decoder skeleton

Nearly every model in the roster is a **decoder-only Transformer**. Structurally
GPT-2 (2019) and DeepSeek V3 (2024) are remarkably similar; the differences are
refinements to the block internals, not a new paradigm. The canonical block with
Pre-Norm + residuals:

```
x = x + Attention(Norm(x))   # attention sub-layer
x = x + FFN(Norm(x))         # feed-forward sub-layer
```

Everything below is a variation on *what goes in those four slots* (the two
Norms, the Attention, the FFN) plus how positions are encoded and how the KV
cache is managed.

---

## 2. Normalization

- **LayerNorm** centers (subtract mean) and scales (divide by std), then applies
  learned gain+bias. **RMSNorm** drops the mean-centering and bias, dividing only
  by the root-mean-square. Cheaper, fewer params, equally stable at scale — the
  modern default (Llama, Qwen, DeepSeek, GLM, Gemma).
- **Placement**: Pre-Norm (normalize *before* the sub-layer, inside the residual)
  is standard and stable. Some models (OLMo 2) use Post-Norm variants. Gemma 3/4
  and Arcee Trinity use a **four-norm** layout (norm before *and* after each
  sub-layer) for extra stability.
- **QK-Norm**: apply RMSNorm to the query and key vectors before computing
  attention scores. Stops attention logits from blowing up; key for stable
  large-scale training. Used by Qwen3, Kimi K2 (as QK-Clip variant), Trinity.
- **Depth-scaled gain**: initialize a block's second norm gain to ~1/√L (L =
  layer count) so early-training residual updates start small and grow. Trinity
  uses this.

## 3. Feed-forward — SwiGLU

Replaces `W2·GELU(W1·x)` with a gated form:
```
SwiGLU(x) = W2 · ( SiLU(W1·x) ⊗ (W3·x) )
```
Two parallel projections (one gated by SiLU, one linear), multiplied
element-wise. More expressive per parameter; universal in the roster. Intermediate
width is typically ~8/3 × d_model to keep parameter count comparable to the older
4× GELU FFN.

## 4. Positional encoding

- **RoPE (Rotary Positional Embedding)**: encodes position as a rotation applied
  to Q and K, so attention sees *relative* position. Extrapolates far better than
  absolute embeddings. Universal.
- **RoPE base frequency / ABF**: the base (default 10,000) sets how fast rotation
  frequencies decay. Raising it (Llama 3.1, DeepSeek V3, Qwen3 → 1,000,000 via
  Adjusted Base Frequency) slows the "blurring" of distant positions, helping
  long context.
- **YaRN**: an inference-time RoPE-frequency scaling that extends usable context
  beyond the trained length (e.g. 4×). Often applied only to global-attention
  layers (OLMo 3, Gemma).
- **DCA (Dual Chunk Attention)**: chunks long sequences at inference to keep
  attention tractable while preserving relative positions across chunks (Qwen3).
- **NoPE (No Positional Embedding)**: omit positional encoding entirely in
  *global* attention layers; the model infers order through the causal mask and
  attention dynamics. Improves length generalization. Used in SmolLM3, Arcee
  Trinity (global layers), Qwen3-Next.

## 5. Attention family (the central fork)

Motivation: the KV cache (stored keys/values for every past token) dominates
long-context memory. The family is a sequence of answers to "how do we shrink the
KV cache without losing quality?"

- **MHA (Multi-Head Attention)** — every query head has its own K and V head.
  Most expressive, biggest KV cache.
- **MQA (Multi-Query Attention)** — all query heads share ONE K/V head. Smallest
  cache, some quality loss. Gemma 4 E2B uses MQA.
- **GQA (Grouped-Query Attention)** — query heads are split into G groups; each
  group shares a K/V head. The pragmatic middle ground and current mainstream
  (Llama, Qwen3 dense, Gemma 4 26B/31B, Trinity). `n_kv = n_q / group_size`.
- **MLA (Multi-Head Latent Attention)** — DeepSeek V2's invention (also in V3,
  R1, Kimi K2). Compress K and V into a low-dim latent vector `c_kv`, store *that*
  in the KV cache, and up-project back to full K/V at compute time. Q is also
  compressed (training-time). Costs an extra matmul; cuts KV cache ~5–10× vs GQA
  and, per DeepSeek's ablations, can match or beat full MHA on quality when tuned.
  DeepSeek V3 specifics: KV-compression dim ≈ 512, Q-compression dim ≈ 1536,
  per-head dim 128.

Mental ranking from the ablations: **MLA ≳ GQA > MQA** on quality; **MQA <
GQA < MLA < MHA** on KV-cache size (MLA's stored latent is tiny).

## 6. Long-context attention variants

- **Sliding-Window Attention (SWA)** — each token attends only to the last *t*
  tokens (e.g. 512–4096), turning O(n²) into ~O(n·t). Used in a ratio with global
  layers: Gemma 3 (5:1 local:global), Trinity/OLMo 3 (3:1), Laguna. Cheap, but
  pure-local layers can't see the whole context — hence the periodic global layer.
- **DeepSeek Sparse Attention (DSA)** — *learned* sparsity instead of a fixed
  window. A **lightning indexer** scores each past token's relevance to the
  current query (reusing MLA's compressed reps): `I_{t,s} = Σ_j w_{t,j}
  ReLU(q_{t,j}·k_s)`. A **token selector** keeps the top-k (k=2048 in released
  code) and masks the rest. Drops attention from O(L²) to O(L·k). Introduced in
  DeepSeek V3.2-Exp; adopted by GLM-5.
- **Lightning / linear attention (MiniMax)** — replace softmax with a kernel
  feature map φ so attention becomes `O_t = (Σ_{s≤t} φ(K_s)ᵀV_s)·φ(Q_t) /
  (Σ φ(K_s)ᵀ)`, an O(n·d²) running sum with a *fixed-size* hidden state (no
  growing KV cache). MiniMax uses a 7:1 lightning:softmax block ratio so a global
  softmax layer periodically restores full mixing. Linear attention has different
  per-layer decay rates (early layers local, later layers global), which
  complicates long-context training (gradient spikes) — handle via staged length
  extension. Kimi Linear (48B) showed linear hybrids can match MLA on long-context
  + reasoning while being much faster.
- **Compressed Convolutional Attention (CCA, Zyphra ZAYA1)** — compress Q/K/V and
  run attention *inside* the compressed latent space (vs MLA which compresses only
  for storage then up-projects). Adds depthwise conv mixing on the compressed Q/K
  (not V) to restore local expressiveness. Cuts both KV cache *and* attention
  FLOPs. Reported to beat MLA at comparable compression.

## 7. KV-cache math and sharing

KV-cache size (bf16) ≈ `2 × n_layers × n_kv_heads × head_dim × seq_len × 2 bytes`.
This is why n_kv_heads (GQA), latent dim (MLA), and layer count all matter, and
why long context is the binding constraint.

- **Cross-layer KV sharing (Gemma 4 E2B/E4B)** — later layers reuse the K/V
  computed by an earlier layer of the *same* attention type (sliding shares with
  sliding, global with global); each layer still computes its own Q. Sharing ~half
  the layers ≈ halves the KV cache (E2B saves ~2.7 GB at 128K). Small capacity hit.
- **Per-layer query-head budgeting (Laguna XS.2)** — vary `num_attention_heads`
  per layer (more Q heads on cheap sliding layers, fewer on expensive global
  layers) while keeping KV heads fixed. Spends attention capacity where it helps.

## 8. Mixture of Experts (MoE)

Replace the FFN with N expert FFNs; a router picks top-K per token:
`FFN_MoE(x) = Σ_{i∈TopK(Router(x))} g_i · Expert_i(x)`.

- **Total vs active params** — capacity scales with total params; compute scales
  with active. DeepSeek V3: 671B total / 37B active (5.5%). Kimi K2: 1.04T / 32B
  (3.2%). Qwen3-235B: 235B / 22B (9.4%). The "act huge, compute small" trick.
- **Granularity** — DeepSeekMoE uses *many small* experts (V3: 256 routed + 1
  shared, top-8) vs Mixtral's *few large* (8 experts, top-2). Finer granularity =
  more routing combinations = more capacity, but more all-to-all communication.
  Coarsening experts improves inference throughput (Mistral 3, Trinity did this).
- **Shared expert** — an always-on expert giving every token a capability floor
  (DeepSeek, Kimi, Llama 4). Qwen3 *dropped* it; no field consensus on whether
  it's worth the compute.
- **Sparsity scaling law (Kimi K2)** — at fixed active params, raising total
  expert count keeps lowering loss → motivated 256 → 384 experts.
- **Load balancing** — naive routing collapses onto a few experts. The classic
  fix is an auxiliary loss (interferes with the main objective). DeepSeek V3's
  **auxiliary-loss-free** scheme adds a per-expert bias to routing logits and
  nudges the bias by observed load — balancing without polluting gradients.

## 9. Multi-Token Prediction (MTP)

Train extra lightweight heads to predict tokens t+1…t+k (DeepSeek recommended
k=4 in the MTP paper; DeepSeek V3 used MTP-1). The extra signal speeds training.
At inference MTP heads enable **speculative decoding**: propose k tokens, verify
with the main model, accept the run that matches → higher throughput at one-token
quality. Step 3.5/3.7 Flash uses MTP-3 at *both* train and inference (unusual);
GLM-4.7 and MiniMax M2.1 also use MTP-3.

## 10. Other efficiency levers

- **FP8 mixed-precision training** — DeepSeek V3 pioneered production-scale FP8,
  cutting memory and interconnect traffic ~2×; key to its low training cost.
  GLM-4.6 ran FP8 on non-NVIDIA (Cambricon) silicon.
- **Quantization (Int4/Int8)** for inference — shrinks the weight footprint;
  essential for fitting MoE total params in memory.
- **Per-Layer Embeddings (PLE, Gemma 4 E2B/E4B)** — store extra token-specific
  capacity in cheap per-layer embedding tables (looked up, gated into each block)
  instead of widening the transformer. Drives the "effective" param count: E2B is
  ~2.3B effective / 5.1B with embeddings.
- **Gated attention output** — multiply the attention output by a learned
  sigmoid gate before the output projection. Reduces *attention sinks* (weight
  piling onto the first tokens) and improves long-sequence stability. Qwen3-Next,
  Trinity, Step 3.5 Flash.

## 11. Post-training

The 2025–26 paradigm shift: capability increasingly comes from post-training, not
just bigger pre-training.

- **RLVR (RL with Verifiable Rewards)** — reward = automatic verification (math
  answer correct, code passes tests). No human preference labels needed; reasoning
  CoT emerges. The DeepSeek R1 breakthrough.
- **GRPO (Group Relative Policy Optimization)** — PPO minus the critic: sample a
  group of responses, use the group's reward mean/std as the baseline for
  advantage. Simpler and cheaper than PPO+critic. `A_i = (R_i − mean(R)) / std(R)`.
- **CISPO (MiniMax M1)** — clips the *importance-sampling weight* rather than the
  per-token update; more stable for linear-attention hybrids. Enabled M1's full RL
  run on 512 H800s in ~3 weeks (~$0.5M).
- **Self-verification + self-refinement (DeepSeekMath V2)** — train an LLM-based
  verifier (not just a symbolic checker) and a proof-generator that uses it as a
  reward model, so the model checks and fixes its own reasoning. Addresses
  "correct answer via flawed reasoning".
- **Strong-to-weak distillation (Qwen3)** — train small models on flagship
  outputs; beats independent small-model RLVR on both quality and cost.
- **Reasoning model types** — *dedicated* (R1), *hybrid* (toggle think/no-think
  via template, e.g. Qwen3, DeepSeek V3.1/V3.2), or *effort-controlled* (system
  prompt sets reasoning effort, e.g. gpt-oss). Note which a model is.

## 12. The analysis lens — what to actually compare

When analyzing any model, isolate the deltas in these slots (ignore marketing):

1. **Attention mechanism** + head/KV config → KV-cache size & quality tradeoff.
2. **MoE config**: total/active, expert count & granularity, shared expert,
   load-balancing → capacity vs compute vs infra cost.
3. **Positional/context strategy**: RoPE base, YaRN/NoPE/DCA, SWA ratio, sparse
   attention → claimed vs *usable* context.
4. **Efficiency stack**: MTP, FP8/quant, KV-sharing, gating, PLE.
5. **Lineage & delta**: nearest neighbor (often the DeepSeek V3 template) and the
   specific changes from it.
6. **Post-training**: reasoning type, RL recipe, distillation.
7. **Economics**: input/output/cached price, effective price, active-params-per-
   dollar, deployment VRAM.

A good analysis answers, for each model: *what did they change, and what
constraint (memory, FLOPs, training stability, cost, context length) was that
change buying them down?*
