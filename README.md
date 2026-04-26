# MagicQuant (v2.0)

**MagicQuant is a benchmark-driven GGUF evaluation and hybrid-discovery system.**  
  
It does not invent new quantization methods.  
It tests them, compares them, mixes them when useful, and **judges what deserves to survive**.  
  
MagicQuant exists to answer a simple question:  
  
> **Which quantized models are actually worth using at each size?**  
  
Most quant releases give you a pile of files, AKA: Q8, Q6, Q5, Q4, and leave you to guess. MagicQuant replaces that guesswork with benchmarks, tensor-group probing, mixed hybrid GGUF builds when they are worth it, and a final survivor list built around meaningful size/fidelity tradeoffs.

---

## What MagicQuant Does

MagicQuant takes the messy quantization space and turns it into a judged survivor list.  
  
It tests standard baselines, learns from external quant strategies, and builds mixed tensor-group hybrids when there may be a better size/fidelity trade hiding between normal quant levels.  
  
Then it validates the results.  
  
MagicQuant does not assume hybrids are better. It does not assume baselines are safe. Every option has to earn its slot.  
  
A final MagicQuant release is meant to show:  
  
* what is smallest  
* what is safest  
* what is meaningfully in-between  
* what was removed as redundant or not worth the damage  
* and what the real benchmark numbers say  
  
If a model survives MagicQuant, it survived because the trade was worth showing.

---

## Example

The following example is Qwen3-4B-2507-Instruct going through MagicQuants pipeline and the final results:

| Name                                                                                      | Provider   | Quant Family |      KLD | Size (GB) |
| ----------------------------------------------------------------------------------------- | ---------- | ------------ | -------: | --------: |
| LM-Q8_0                                                                                   | llama.cpp  | Q8_0         | 0.001339 |      3.99 |
| MQ-Q6_K_1                                                                                 | MagicQuant | Q6_K         | 0.001817 |      3.58 |
| UD-Q6_K_XL                                                                                | Unsloth    | UD-Q6_K_XL   | 0.002111 |      3.41 |
| LM-Q6_K                                                                                   | llama.cpp  | Q6_K         | 0.004640 |      3.08 |
| [<u>MQ-Q5_K_1</u>](#winner-notes "Replaced: MQ-Q5_K")                                     | MagicQuant | Q5_K         | 0.006632 |      2.88 |
| [<u>UD-Q5_K_XL</u>](#winner-notes "Replaced: LM-Q5_K, LM-Q5_K_S")                         | Unsloth    | UD-Q5_K_XL   | 0.009839 |      2.73 |
| [<u>MQ-Q4_K_M_1</u>](#winner-notes "Replaced: MQ-Q4_K_M, UD-Q4_K_XL, LM-Q4_K_M + 1 more") | MagicQuant | Q4_K_M       | 0.020346 |      2.44 |
| [<u>LM-Q4_K_S</u>](#winner-notes "Replaced: LM-IQ4_NL")                                   | llama.cpp  | Q4_K_S       | 0.029803 |      2.22 |
| LM-IQ4_XS                                                                                 | llama.cpp  | IQ4_XS       | 0.031300 |      2.11 |
| UD-Q3_K_XL                                                                                | Unsloth    | UD-Q3_K_XL   | 0.072278 |      1.98 |

The table above includes a mix of standard llama.cpp quantizations, Unsloth Dynamic GGUF models, and MagicQuant hybrids.

In some cases, dominance is absolute. For example, Unsloth’s **Q5_K_XL** fully replaces the standard llama.cpp **Q5_K**, as MagicQuant determined the baseline offered no meaningful tradeoff in comparison.

More interesting are the hybrid outcomes. **MQ-Q4_K_M_1** emerged as a clear dominant variant, replacing multiple candidates simultaneously (_UD-Q4_K_XL, MQ-Q4_K_M, LM-Q4_K_M_). While baseline quants can sometimes achieve similar dominance, this case highlights a hybrid configuration that decisively outperformed across the board.

**MQ-Q5_K_1** is another notable result. It leverages Unsloth’s learned tensor behavior (_Q5_K_XL_) within the `ffn_up_gate`, discovering a middle ground between **UD-Q5_K_XL** and **LM-Q6_K**. The result is a hybrid that achieves a disproportionately large KLD improvement relative to the additional size cost, exceeding a simple linear tradeoff.

The table below breaks down these MagicQuant hybrids by tensor group, showing the assigned quantization for each, whether derived from llama.cpp baselines or Unsloth’s learned tensor mappings.

| Name        | embeddings | attn_q | attn_kv | attn_output | ffn_up_gate | ffn_down |
| ----------- | ---------- | ------ | ------- | ----------- | ----------- | -------- |
| MQ-Q6_K_1   | Q8_0       | Q8_0   | Q8_0    | Q8_0        | Q6_K        | Q8_0     |
| MQ-Q5_K_1   | Q8_0       | Q5_K   | Q8_0    | Q6_K        | UD-Q5_K_XL  | Q5_K_S   |
| MQ-Q4_K_M_1 | Q8_0       | Q5_K   | Q8_0    | Q6_K        | IQ4_XS      | IQ4_XS   |

---
## Deeper Understanding

**Deep Dive Documentation**:
* [Bad Trades, Early Pruning, and Search-Space Collapse](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/Bad-Trades-And-Early-Pruning.md)
* [Learning From Existing Quantizations (Tensor Configuration Sources)](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/Learning-From-Existing-Quantizations)
* [Nonlinear Winners and Survivor Selection](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/Nonlinear-Winners-And-Survivors)
* [Prediction Engine](https://github.com/magiccodingman/MagicQuant-Wiki/blob/main/docs/Prediction-Engine)

When you see a MagicQuant hybrid, it’s not just a “Q4.5” sitting somewhere between Q4 and Q5. It represents a discovered configuration where the **KLD reduction is non-linear relative to the size increase**, a genuinely better trade space. Not universally “better” than everything else, but a variant that earned its place through measurable advantage.

Whether the winner is a hybrid or a pure baseline from llama.cpp or Unsloth, any quant that removes another from the final selection does so because its dominance made the alternative no longer worth considering.

The goal is not to flood the space with near-duplicates offering negligible KLD gains for minimal size differences, nor to claim superiority for the sake of it. In fact, that’s explicitly what MagicQuant avoids.

MagicQuant is built around transparency, honesty, maintainability, and most importantly trust. As it evaluates new architectures and quant families, it doesn’t invent quantization schemes in isolation. Instead, it learns from proven tensor assignments provided by trusted sources like llama.cpp and Unsloth. If those baselines are stable, MagicQuant operates within that same safe space, extending rather than reinventing.

That said, the system is designed to adapt. Edge cases can exist, but the architecture is intentionally flexible to handle them.

Finally, everything, from how winners are selected to how the prediction system works, is fully documented. The intent is for MagicQuant to be understandable, reproducible, and above all, trustworthy.

### How MagicQuant Works

```
        ┌────────────────────────────┐
        │   Input Quantized Models   │
        │ ───────────────────────── │
        │ llama.cpp / Unsloth / etc │
        └────────────┬──────────────┘
                     │
                     │ Inspect tensors
                     ▼
        ┌────────────────────────────┐
        │ Tensor Extraction Layer    │
        │ ───────────────────────── │
        │ - Read all tensors         │
        │ - Detect quant types       │
        │ - Capture F32 / BF16       │
        └────────────┬──────────────┘
                     │
                     │ Group by role
                     ▼
        ┌────────────────────────────┐
        │ Tensor Group Mapping       │
        │ ───────────────────────── │
        │ embeddings                │
        │ attn_q / attn_kv / output │
        │ ffn_up_gate / ffn_down    │
        │ lm_head / moe_*           │
        └────────────┬──────────────┘
                     │
                     │ Learn configs
                     ▼
        ┌────────────────────────────┐
        │ Learned Config Library     │
        │ ───────────────────────── │
        │ "Q5_K attn_q pattern"     │
        │ "UD-Q5_K_XL ffn pattern"  │
        │ etc                       │
        └────────────┬──────────────┘
                     │
                     │ Normalize external configs
                     ▼
        ┌────────────────────────────┐
        │ Controlled Rebuild Layer   │
        │ ───────────────────────── │
        │ - Apply configs to BF16    │
        │ - Use MagicQuant imatrix   │
        │ - Equal comparison ground  │
        └────────────┬──────────────┘
                     │
                     │ Feed into
                     ▼
        ┌────────────────────────────┐
        │ Hybrid Construction Engine │
        │ ───────────────────────── │
        │ Mix tensor groups across   │
        │ learned configurations     │
        └────────────┬──────────────┘
                     │
                     │ Evaluate candidates
                     ▼
        ┌────────────────────────────┐
        │ Prediction + Isolation     │
        │ ───────────────────────── │
        │ - Group-level testing      │
        │ - Rank-safe prediction     │
        └────────────┬──────────────┘
                     │
                     │ Build real GGUF
                     ▼
        ┌────────────────────────────┐
        │ Benchmark Layer            │
        │ ───────────────────────── │
        │ - KLD (primary)            │
        │ - PPL (secondary)          │
        └────────────┬──────────────┘
                     │
                     │ Final decision
                     ▼
        ┌────────────────────────────┐
        │ Survivor Selection         │
        │ ───────────────────────── │
        │ - Dominance pruning        │
        │ - Nonlinear winners        │
        │ - Spacing collapse         │
        └────────────────────────────┘
```