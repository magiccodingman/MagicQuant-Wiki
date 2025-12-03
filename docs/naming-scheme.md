# **MagicQuant Hybrid Naming Scheme (Short & Practical Guide)**

MagicQuant produces hybrid quantization models where different parts of the network receive different quant schemes (e.g., MXFP4 base with BF16 embeddings, IQ4_NL attention layers, etc.).

A full tensor-by-tensor representation would be unreadable as a filename, so MagicQuant uses a **compact, modular naming scheme**:

---

## **1. Base Format**

Every name begins with the model and its **base quantization**:

```
<ModelName>-<BaseQuant>
```

Examples:

```
Qwen3-4B-MXFP4
Qwen3-7B-IQ4_NL
```

This tells users the *default* quantization for the majority of tensors.

---

## **2. Only Deviations Are Listed**

If certain tensor groups use a different quant scheme than the base, they appear afterwards as:

```
<GroupLetters>-<Quant>
```

Multiple group blocks can be chained:

```
<Model>-<Base>-<Groups>-<Quant>-<Groups>-<Quant>...
```

Only *exceptions* appear.
Anything not listed = uses the base quant.

---

## **3. Group Abbreviations**

These are the compact codes for each major tensor group:

| Code  | Tensor Group        | Meaning                        |
| ----- | ------------------- | ------------------------------ |
| **E** | Embeddings          | token_embd.weight              |
| **H** | LM Head             | output.weight / lm_head.weight |
| **Q** | Attention Query     | attn_q.weight                  |
| **K** | Attention Key/Value | attn_k.weight + attn_v.weight  |
| **O** | Attention Output    | attn_output.weight             |
| **U** | FFN Up/Gate         | ffn_up + ffn_gate              |
| **D** | FFN Down            | ffn_down.weight                |
| **X** | MoE Experts         | ffn_*_expert.*                 |
| **R** | MoE Router          | router.*, gate.*               |

> **You suggested “K” instead of “KV,” and yes — the shorter the better.**
> “K” represents the entire Key+Value branch (people will get it).

---

## **4. Compressing Groups With the Same Quantization**

If multiple groups share the same quant scheme, combine them:

```
EH-B16
```

Means:

* Embeddings = BF16
* Head = BF16

Another block:

```
QKO-IQ4NL
```

Means:

* Q, K, O → IQ4_NL

You can stack as many blocks as needed.

---

## **5. Complete Example**

### **Original Long-Form Description**

* Base: MXFP4
* Embeddings → BF16
* LM Head → BF16
* Q, K, O → IQ4_NL
* Everything else → MXFP4

### **Hybrid Name:**

```
Qwen3-4B-MXFP4-EH-B16-QKO-IQ4NL.gguf
```

This reads as:

* “Model starts as MXFP4”
* “Embeddings + Head upgraded to BF16”
* “Attention Q, K, O moved to IQ4_NL”
* “Everything else = MXFP4”

Clean. Simple. Understandable. Portable.

---

## **6. If Only One Group Changes?**

Example: Only embeddings become Q5_K.

```
Qwen3-4B-MXFP4-E-Q5K.gguf
```

If only MoE router changes:

```
Qwen3-4B-MXFP4-R-Q8.gguf
```

---

## **7. If EVERYTHING is the base (pure baseline)**

Just:

```
Qwen3-4B-Q4_K_M.gguf
```

No extra suffixes.

---

## **8. Notation Guidelines (For Clarity and Aesthetics)**

* Use hyphens `-` throughout the naming scheme for consistency.
* No need for `_` unless the quant type requires it (`IQ4_NL`, `Q4_K_M`, etc.).
* Order of groups doesn’t matter, but a consistent order is recommended:

**E, H, Q, K, O, U, D, X, R**

This mirrors information flow: embeddings → attention → ffn → moe.

---

# **Final Example Set (For Quick Visual)**

| Description                                | Final Name                     |
| ------------------------------------------ | ------------------------------ |
| Base MXFP4, only embeddings = BF16         | `MXFP4-E-B16`                  |
| Base IQ4_NL, Q/K/O = Q6_K                  | `IQ4NL-QKO-Q6K`                |
| Base Q6_K, Head & Router = BF16            | `Q6K-HR-B16`                   |
| Everything BF16 → no hybrid                | `B16` (or `F16/BF16` baseline) |
| Full MoE override: experts + router = Q8_0 | `X R-Q8_0` → `XR-Q8_0`         |
