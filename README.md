# MagicQuant Wiki
> Evolution process to find the best quant tensor weights to build the most optimal GGUF options for an AI model.


### Index

* [Naming Scheme](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/naming-scheme.md) - Learn about the MagicQuant naming scheme which helps shorten lengthy names.
* [Precision Loss Guide](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/precision-loss-guide.md) - A guide to understanding precision loss and the philosophy behind this project.
* [Model Selection & Philosophy](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/model-selection.md) - Why and what models are uploaded.
* [HuggingFace: Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant) - The best of the best quants.

---

## Evolutionary Tensor Search: A Hybrid Quantization Framework for Optimal LLM Compression

### Abstract

Current quantization methodologies typically apply global precision schemes (e.g., Q4, Q8) uniformly across model weights. This approach ignores the heterogeneous sensitivity of Transformer architectures, where specific layers (e.g., Embeddings, Attention Heads) are disproportionately sensitive to quantization noise compared to others (e.g., FFN blocks). This paper introduces **"Magic Quant,"** an evolutionary search framework. By dynamically grouping tensors, probing layer sensitivity, and employing a residual-learning predictor, the system identifies hybrid quantization combinations that significantly outperform standard baselines in the Size/Perplexity/Speed trade-off space.

### 1\. Introduction

The dogma of modern quantization suggests choosing a preset format (like GGUF’s Q4\_K\_M) and applying it. However, this is locally optimal at best. A "Q4" model might have 90% of its parameters in FFN layers that could withstand Q2 compression, while its attention heads break down at anything less than Q6.
This framework proposes a shift from **Global Quantization** to **Functional Group Mixed-Precision Quantization**. By automating the discovery of these mixtures, we achieve "impossible" quants—models that are smaller than a standard Q4 but retain the perplexity of a Q5.

### 2\. Methodology

#### 2.1. Tensor Grouping Strategy

Rather than optimizing per-layer (which creates a search space of $N_{schemes}^{N_{layers}}$), the system groups tensors by architectural role. This reduces dimensionality while retaining functional granularity.

  * **Groups:** `embeddings`, `lm_head`, `attn_q`, `attn_kv`, `attn_output`, `ffn_up_gate`, `ffn_down`.
  * **MoE Specifics:** If a Mixture-of-Experts architecture is detected, the system isolates `moe_router` (highly sensitive) from `moe_experts` (highly robust).

#### 2.2. The Sensitivity Probe Phase ("Feeling Out the Tensors")

Before the search begins, the system executes a "Probe Phase" to model the model's physical response to compression.

1.  **Baseline:** Measures the uncompressed (F16/BF16) perplexity.
2.  **Aggressive Probing:** For each group, the system creates a temporary hybrid where *only* that group is crushed to a low precision (e.g., MXFP4) while the rest remain at baseline.
3.  **Sensitivity Weighting:** The degradation in perplexity ($\Delta PPL$) is measured. Groups that cause high degradation are assigned high sensitivity weights; robust groups are assigned lower weights.

#### 2.3. The Predictive Scoring Model

To avoid measuring millions of combinations, a predictive model estimates the performance of a theoretical combination $C$.

**The Loss Prediction:**

$$Loss_{pred} = \sum_{g \in Groups} (S_g \times W_g)$$

Where $S_g$ is the sensitivity of the group and $W_g$ is the quantization noise factor of the chosen scheme (e.g., Q4 vs Q8).

**The Non-Linear Collapse Penalty:**
The system enforces a "Physics" constraint. If multiple "Brain Layers" (Embeddings + Output Head + Attention Output) are simultaneously compressed to aggressive schemes, the model predicts a non-linear degradation (Collapse).

$$if \ Count(Brain_{crushed}) \ge 2: Loss_{final} = max(Loss_{pred} \times \alpha, Loss_{pred} + \beta)$$

**Active Learning (Residual Corrections):**
As real measurements come in, the system compares $Loss_{measured}$ vs $Loss_{predicted}$. It calculates residuals pattern-keys (e.g., `Base=MXFP4 | Embed=Q8`). These residuals are fed back into the predictor, allowing it to "learn" that specific combinations yield better or worse results than the theoretical math suggests.

#### 2.4. Evolutionary Survival Rounds

The search process is iterative (Survival Rounds), mimicking natural selection:

1.  **Generation:** A massive pool of candidates is generated based on valid schemes (BF16, Q8, Q6, Q5, IQ4, MXFP4).
2.  **Classification:** Candidates are sorted into "Tiers" based on standard baselines. (e.g., "Tier Q4" contains all hybrids smaller than a Q5 but larger than a Q3).
3.  **Selection (The Tournament):** Within each tier, winners are chosen for:
      * *Lowest Loss* (Precision Winner)
      * *Smallest Size* (Efficiency Winner)
      * *Highest TPS* (Speed Winner)
      * *Balanced Score* (Weighted average of all three)
4.  **Mutation:** Top winners undergo mutation:
      * *Protector Strategy:* The most sensitive layer is upgraded (e.g., Q6 $\to$ Q8).
      * *Crusher Strategy:* The least sensitive layer is downgraded (e.g., Q4 $\to$ MXFP4).

#### 2.5. Epsilon-Greedy Exploration

To prevent getting stuck in local optima (exploiting known good configs), the system employs an $\epsilon$-greedy strategy. In every round, a random sample of "Predicted Losers" is sent to the measurement phase. If a predicted loser turns out to be a "Surprise Winner" (better than predicted), the Residual Correction engine updates, shifting the entire search direction toward that successful pattern.

### 3\. Experimental Setup

  * **Context:** All benchmarks use a massive **32k context window**, stressing the KV cache and attention layers far more than standard 4k/8k benchmarks.
  * **Perplexity:** Measured across three distinct domains to ensure generalization:
    1.  *General* (Wikitext)
    2.  *Code* (CodeParrot)
    3.  *Math* (GSM8k)
  * **Inference Speed:** Tokens Per Second (TPS) is measured to ensure that complex compression schemes (like IQ4\_NL) do not bottleneck inference decoding.

### 4\. Conclusion

The "Evolutionary Tensor Search" algorithm demonstrates that optimal quantization is not a static choice but a dynamic search problem. By decoupling tensor groups and applying evolutionary pressure, the system consistently finds hybrid configurations that dominate the standard quantization Pareto frontier.

-----

# Technical Architecture & Methodology

## System Overview: The "Magic Quant" Pipeline

The Magic Quant framework operates as an automated research agent. Instead of treating model quantization as a static format conversion, it treats it as a high-dimensional search problem. The system employs an **Active Learning Evolutionary Strategy** to navigate the trade-off space between Model Size, Inference Speed (TPS), and Perplexity (PPL).

The pipeline consists of four distinct phases that cycle until convergence: **Initialization**, **Sensitivity Probing**, **Evolutionary Search**, and **Validation**.

### 1. Phase I: Structural Analysis & Initialization
Before optimization begins, the system constructs a "mental model" of the target LLM.

* **Tensor Topology Scanning:** The system inspects the raw GGUF structure to map specific tensors to functional groups. It identifies architectural components such as *Embeddings*, *Attention Heads (Q/K/V)*, *Output Projections*, and *Feed-Forward Networks (FFN)*. It automatically detects specialized architectures like Mixture-of-Experts (MoE) or Tied Embeddings.
* **Baseline Calibration:** To establish "Ground Truth," the system builds and benchmarks standard reference points (e.g., FP16, Q8_0, Q6_K, Q4_K_M). These serve as the boundaries for the **Tier System**, ensuring that any discovered hybrid is strictly classified relative to known standards (e.g., "Tier Q4" = larger than Q3, smaller than Q5).
* **Seed Injection (Warm Start):** The search is seeded with high-probability "human intuition" heuristics. For example, in MoE models, it injects "High-Contrast" seeds (High-Precision Router / Low-Precision Experts) to force the algorithm to evaluate these strategies early.

### 1.5. Dynamic Weighting and Modeling Inputs

Immediately following the Tensor Topology Scanning, the system calculates two critical sets of weights that govern the entire predictive process:

* **1. Physics Weights (Size/TPS Factor):** The system scans the unquantized GGUF to count the total number of parameters belonging to each functional group.

    * $$W_{physics, g} = \frac{\text{Parameter Count}_g}{\text{Total Model Parameters}}$$
    
    * These weights sum to 1.0 and are used to predict the model's final **size** and **TPS**. They are crucial because the FFN group might account for 80% of $W_{physics}$, meaning its compression factor heavily influences the final model size.
    
    * The prediction verifies the **physical integrity** of the hybrid configuration before performance is estimated.

* **2. Sensitivity Weights (Loss Factor):** These weights, determined in Phase II, measure the *quality impact* of each group.
    * $$W_{sensitivity, g} = \frac{\Delta PPL_{probe, g}}{\sum \Delta PPL_{probes}}$$
    
    * These weights govern the **Loss Prediction** and identify groups where every bit of precision counts (e.g., Attention Q/K/V layers).

### 2. Phase II: The Sensitivity Probe
This phase is unique to Magic Quant. Before generating hybrids, the system "feels out" the model’s electrical sensitivity to compression.

* **The Probe Strategy:** The system generates temporary test models where a single functional group (e.g., `attn_output`) is aggressively compressed (e.g., to `MXFP4`) while the rest of the model remains at high precision.
* **Sensitivity Weighting:** By measuring the perplexity degradation ($\Delta PPL$) of these probes, the system assigns a **Sensitivity Score** to each group.
    * *High Sensitivity:* Layers that cause massive brain damage when crushed (typically Embeddings or Attention Output).
    * *Low Sensitivity:* Layers that show minimal degradation (typically FFN or MoE Experts).
    * These weights inform the predictive engine, allowing it to prioritize "protecting" sensitive layers during hybrid generation.

### 3. Phase III: The Evolutionary Core (The Engine)
This is the iterative loop where the system discovers optimal configurations. It follows a **Predict $\rightarrow$ Measure $\rightarrow$ Learn** cycle.

#### A. The Predictive Model
Rather than measuring millions of combinations physically, the system uses a simulator to estimate the performance of a theoretical hybrid.
* **Loss Prediction:** Calculated using the **Learned Sensitivity Weights** derived from Phase II, combined with the quantization noise factors of the chosen formats.
* **Non-Linear Collapse Penalty:** The system simulates "Physics Constraints." It recognizes that while compressing *one* sensitive layer might be survivable, compressing *multiple* sensitive layers (e.g., Embeddings + Output Head) simultaneously often causes non-linear model collapse. A penalty function artificially inflates the predicted loss for these combinations, effectively pruning them from the search space.
* **Balanced Scoring:** Candidates are ranked by a composite objective function:
    $$Score = (0.4 \times Precision) + (0.3 \times Size) + (0.3 \times Speed)$$

#### B. The Selection Tournament (Measurement)
Due to the computational cost of benchmarking, the system is extremely frugal with real measurements. It selects candidates for the "Real World" based on:
1.  **Tier Winners:** The statistically best predicted combo for every Tier (Q6, Q5, Q4, etc.).
2.  **Ambiguity Resolution:** If two combos have near-identical predicted scores, both are measured to resolve the tie.
3.  **Epsilon-Greedy Exploration:** To avoid local optima, the system randomly samples "Predicted Losers." If a "Loser" performs unexpectedly well, it triggers a massive update to the learning model.
4.  **Strict Dominance Filtering:** Any hybrid that is physically as large as a `Q8_0` file is subjected to a "Brutal Filter"—it is immediately discarded unless it strictly dominates the baseline in both precision and speed.

#### C. Adaptive Mutation Strategies
When a successful hybrid is identified (a "Survivor"), the system spawns mutant variants to push the Pareto frontier further:
* **The Protector Strategy:** Identifies the layer with the highest sensitivity score that is *not* yet max precision and upgrades it (e.g., `Attn_Output: Q6` $\rightarrow$ `Q8`).
* **The Crusher Strategy:** Identifies the layer with the lowest sensitivity score (the "fat") and downgrades it to squeeze out size savings (e.g., `FFN: Q4` $\rightarrow$ `MXFP4`).

### 4. Phase IV: Active Learning & Calibration
This is the feedback mechanism that makes the system "smart."
* **Residual Analysis:** After every real-world measurement, the system compares the *Predicted Loss* vs. the *Actual Loss*.
* **Pattern Correction:** It identifies specific patterns causing prediction errors (e.g., *"This architecture reacts poorly to MXFP4 Embeddings regardless of what the math says"*).
* **Global Calibration:** The system applies a correction factor to all future predictions containing that pattern. Over several rounds, the predictor becomes highly accurate at forecasting the specific quirks of the model architecture.

### 5. Benchmark Validation
No metric is trusted unless verified. The system runs a rigorous validation suite on every measured candidate:
* **Inference Speed:** Measured in Tokens Per Second (TPS) to ensure that complex quantization mixes (like `IQ4_NL`) do not introduce decoding bottlenecks.
* **Tri-Domain Perplexity:** Precision is measured across three distinct datasets to ensure the model hasn't overfit to a specific domain:
    1.  **General Language** (Wikitext)
    2.  **Coding Capabilities** (CodeParrot)
    3.  **Mathematical Reasoning** (GSM8k)

### 6. Special Handling: Mixture of Experts (MoE)
The system includes specialized logic for MoE architectures (e.g., Mixtral, Qwen-MoE).
* **Router Prioritization:** The system treats the `moe_router` (Gating) tensors as high-criticality components, often locking them to higher precision.
* **Expert Compression:** It recognizes `moe_experts` as highly sparse and robust, frequently targeting them for aggressive `MXFP4` or `IQ4` quantization strategies while maintaining a high-precision "skeleton" for the rest of the model.

---

# Empirical Findings & The "MXFP4 Anomaly"

### 3.1. The "MXFP4" Phenomenon
One of the most significant discoveries made by the **Evolutionary Tensor Search** is the recurring dominance of the `MXFP4_MOE` (Microscaling Float 4) quantization format as a *base layer*, even for dense (non-MoE) models.

In standard quantization literature, `MXFP4` is often viewed as a niche format optimized for specific hardware or exclusively for Mixture-of-Experts architectures. However, the evolutionary algorithm frequently converges on a specific architectural pattern:
1.  **The Body (Base):** The vast majority of parameters (Feed-Forward Networks) are compressed to `MXFP4_MOE`.
2.  **The Brain (Outliers):** A small percentage of sensitive parameters (Embeddings, Attention Output, Routers) are held at `Q8_0` or `Q6_K`.

This "Carbon Fiber Body, Ferrari Engine" approach yields models that consistently break the traditional Pareto frontier, offering lower perplexity than Q5/Q6 models while maintaining sizes comparable to Q4.

### 3.2. Case Study: Mitigating "Brain Damage" in Small Models (Granite 4.0 350M)
Small models (<1B parameters) are notoriously fragile; they lack the redundancy to absorb quantization noise. A standard `MXFP4` conversion usually results in catastrophic "brain damage" (total loss of coherence).

**The Anomaly:**
The Magic Quant system discovered that the *base* weights of Granite 350M are robust enough for MXFP4, provided the specific "Brain Layers" (identified via sensitivity probing) are protected.

| Metric | Standard MXFP4_MOE | Magic Quant Hybrid (MXFP4 Base) | Improvement |
| :--- | :--- | :--- | :--- |
| **Avg PPL Loss** | 1172.27% (Incoherent) | **0.0816%** (Near Lossless) | **~14,000x** |
| **File Size** | 0.34 GB | 0.54 GB | Slightly Larger |
| **TPS** | ~1700 | ~1705 | Equivalent |

*Significance:* The system autonomously salvaged a "dead" quantization format, transforming it from unusable garbage into a SOTA-class compressed model.

### 3.3. Domination of Standard Tiers (Qwen3 4B)
For mid-sized dense models, the system demonstrates that hybrids can strictly dominate standard K-quants. The algorithm discovered a hybrid configuration that is **smaller, faster, and smarter** than the standard `Q5_K`.

**Comparison vs. Standard Q5_K:**

| Configuration | Size (GB) | Speed (TPS) | PPL Loss (Lower is Better) |
| :--- | :--- | :--- | :--- |
| **Standard Q5_K** | 2.69 | 361.03 | 0.5973% |
| **Hybrid (MXFP4 akv_Q6_K-ao_Q5_K-aq_Q5_K-emb_Q6_K-fd_Q5_K-fug_Q5_K)** | **2.65** | **356.70** | **1.15%** |
| **Hybrid (MXFP4 Base - akv_BF16-ao_Q8_0-aq_Q6_K-emb_Q6_K-fd_Q8_0-fug_Q8_0)** | **3.98** | **403.80** | **0.0826%** |

*Note: While the 2.65GB hybrid shows higher loss here, the algorithm also found a 3.98GB variant that achieves 0.08% loss effectively BF16 quality—while still being significantly faster (403 TPS) than the Q5_K baseline (361 TPS).*

This defies the standard "Size vs. Quality" trade-off. By utilizing `MXFP4` for the heavy FFN layers and `Q8_0` for the attention mechanism, we achieve higher throughput (due to memory bandwidth savings on the bulk) without the precision penalty usually associated with 4-bit formats.

### 3.4. Architectural Adaptability (Apriel 1.5 15B)
The evolutionary algorithm does not simply "force" MXFP4 everywhere. It adapts to the specific topology of the neural network. In the case of Apriel 15B, the system detected an unusual sensitivity in specific tensors, leading it to select `Q5_K` for certain layers—a format almost never preferred by other architectures.

**Selected Survivors (Apriel 15B):**
* `mxfp4_moe-te_Q5_K-out_Q5_K-rt_Q5_K-gt_Q5_K` (0.148% Loss)
* `mxfp4_moe-te_IQ4_NL-out_IQ4_NL-rt_IQ4_NL-gt_Q5_K` (0.277% Loss)

This adaptability proves that the "Magic Quant" system is not applying a heuristic (e.g., "always quantize FFN to Q4"), but is genuinely *measuring* the distinct electrical properties of the model's weights.

### 3.5. Resources & Models
The models referenced in this research, along with thousands of other discovered hybrids, are available for public analysis and deployment.

* **Verified "Magic Quant" Survivors (Best of the Best):**
    [HuggingFace: Magic Quant Collection](https://huggingface.co/collections/magiccodingman/magic-quant)

* **Experimental MXFP4 Hybrids (Not to use for production, purely research data):**
    [HuggingFace: MXFP4 Hybrid Collection](https://huggingface.co/collections/magiccodingman/mxfp4-hybrid-gguf)
