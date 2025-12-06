# **MagicQuant Philosophy & Model Selection**


> MagicQuant is not the hybrid.
> 
> MagicQuant is the verdict.

MagicQuant isn’t just a quantizer.
It’s a **belief system** about how quantization *should* be done—and a giant middle finger to the idea of “blindly posting GGUF models because that’s just what people do.”

Every upload has a reason.
And every omission?
That’s intentional too.

MagicQuant posts only **what deserves to survive.** Nothing more, nothing less.

---

# **Why Not All Models Work With MagicQuant**

Not every model bends the knee to hybrid quantization.

Some architectures quantize beautifully with standard baselines and leave very little frontier to push.
Some models have quirks—activation distribution weirdness, MoE gating sensitivity, KV bottlenecks—that make hybrids either unhelpful or flat-out worse.

And sometimes?
The model is simply *already so good* at standard quantization that any attempt to push further produces a downgrade instead of a breakthrough.

MagicQuant will never upload something that is:

* Bigger **and** slower
* Lower precision **and** higher loss
* A strict downgrade to an existing baseline
* Inferior in *every dimension* to another available quant

Because that defeats the entire purpose of the project.

MagicQuant exists to **surface the strongest quantizations each model can offer**, not to spam every combination under the sun.

---

# **Case Study 1: Qwen3 30B A3B Instruct 2507**

📌 Model link:
[https://huggingface.co/magiccodingman/Qwen3-30B-A3B-Instruct-2507-unsloth-MagicQuant-Hybrid-GGUF](https://huggingface.co/magiccodingman/Qwen3-30B-A3B-Instruct-2507-unsloth-MagicQuant-Hybrid-GGUF)

This model is a perfect showcase of the MagicQuant philosophy.
Here were the produced models:

| model                                   | size  | TPS    | loss    |
| --------------------------------------- | ----- | ------ | ------- |
| iq4_nl-EHQKOUD-Q8_0                     | 30.25 | 99.68  | 0.0771% |
| Q5_K                                    | 20.23 | 117.37 | 0.2007% |
| mxfp4_moe-H-B16-EUR-IQ4NL-KO-Q5K-QD-Q6K | 18.93 | 110.54 | 0.3929% |
| IQ4_NL                                  | 16.26 | 138.69 | 0.4198% |
| iq4_nl-EHQKOUD-IQ4NL                    | 16.04 | 149.76 | 2.6323% |

MagicQuant only posts hybrids worth considering—and the baselines below explain why:

| model     | size  | TPS    | loss    |
| --------- | ----- | ------ | ------- |
| BF16      | 56.90 | 44.48  | 0.0000% |
| Q8_0      | 30.25 | 95.03  | 0.0771% |
| Q5_K      | 20.23 | 117.37 | 0.2007% |
| Q6_K      | 23.37 | 108.10 | 0.3089% |
| IQ4_NL    | 16.26 | 138.69 | 0.4198% |
| Q4_K_M    | 17.28 | 132.46 | 1.4766% |
| MXFP4_MOE | 15.15 | 138.34 | 9.0818% |

### **What stands out immediately?**

* Q6_K loses to Q5_K
* Q4_K_M loses to IQ4_NL
* IQ4_NL beats Q4_K_M *even though it "shouldn’t"*
* MXFP4 baseline is unusable here (~9% loss)

This is why MagicQuant never blindly posts:

❌ “Q6_K is always better than Q5_K”
❌ “Q4_K_M is always better than IQ4_NL”
❌ “MXFP4_MOE is always magical”

Because those rules break constantly in the real world.

In this model?

* The “Q8_0 replacement” is **iq4_nl-EHQKOUD-Q8_0**
* The best balanced baselines are **Q5_K** and **IQ4_NL**
* The other hybrids exist for niche use but are not recommended as defaults

MagicQuant uploads *only what’s meaningful*—not every Frankenstein experiment.

---

# **Case Study 2: Qwen3 4B Instruct 2507**

📌 Model link:
[https://huggingface.co/magiccodingman/Qwen3-4B-Instruct-2507-Unsloth-MagicQuant-Hybrid-GGUF/](https://huggingface.co/magiccodingman/Qwen3-4B-Instruct-2507-Unsloth-MagicQuant-Hybrid-GGUF/)

MagicQuant tried many hybrids.
Only one baseline survived: **IQ4_NL**.

Here were the hybrids produced:

| hybrid                                       | size | TPS    | loss    |
| -------------------------------------------- | ---- | ------ | ------- |
| mxfp4_moe-K-B16-QO-Q6K-EUD-Q8_0              | 3.98 | 373.48 | 0.0533% |
| mxfp4_moe-O-Q5K-EQKUD-Q6K                    | 3.03 | 428.37 | 0.1631% |
| mxfp4_moe-QUD-IQ4NL-KO-MXFP4-E-Q8_0          | 2.28 | 411.49 | 0.7356% |
| mxfp4_moe-K-B16-QU-IQ4NL-O-MXFP4-E-Q5K-D-Q6K | 2.62 | 467.79 | 0.8322% |
| IQ4_NL                                       | 2.23 | 426.86 | 0.8996% |
| mxfp4_moe-EQUD-IQ4NL-KO-MXFP4                | 2.10 | 518.15 | 2.0904% |

And the baselines:

| model     | size | TPS    | loss    |
| --------- | ---- | ------ | ------- |
| BF16      | 7.50 | 254.70 | 0.0000% |
| Q8_0      | 3.99 | 362.48 | 0.0724% |
| Q6_K      | 3.08 | 397.92 | 0.2492% |
| Q5_K      | 2.69 | 385.17 | 0.7920% |
| IQ4_NL    | 2.23 | 426.86 | 0.8996% |
| Q4_K_M    | 2.33 | 377.19 | 0.9376% |
| MXFP4_MOE | 2.00 | 467.13 | 8.2231% |

### **This model was almost entirely conquered**

MagicQuant hybrids surpassed or matched:

* Q8_0 in loss
* Q6_K in loss and TPS
* Q5_K in TPS
* IQ4_NL in both directions
* MXFP4 baseline by orders of magnitude

This is MagicQuant’s sweet spot:
**Hybrids that genuinely expand the frontier over all baseline quantizations.**

---

# **The MagicQuant Philosophy**

MagicQuant is built on three guiding principles:

---

### **1. Numbers Over Hype**

No quant is “better” because someone said so.
Only the data matters:

* TPS
* file size
* precision loss

If a quant loses in all three categories, it does not get published, period.

---

### **2. The Model Decides the Rules**

Every architecture is a chaotic little gremlin with its own personality.

Some trends hold:

* Q8_0 ≈ near-lossless
* Q6_K is usually the sweet spot
* Q4_K_M is usually better than IQ4_NL

…until they don’t.

MagicQuant exists to *discover when the rules break*—and then publish only the winners.

---

### **3. MagicQuant’s Promise: Only Post What Deserves to Survive**

You will never find:

* a worse Q4_K_M than IQ4_NL
* a Q6_K that loses to Q5_K (unless MagicQuant warns you)
* an MXFP4 with catastrophic loss
* a hybrid that is strictly inferior to a baseline

MagicQuant is not a dumping ground.
It is a **curated** repository of quantizations that earned the right to exist.

---

# **The Goal of MagicQuant**

MagicQuant is not a “hybrid-only” project.

MagicQuant is a **best-of-the-best** project.

If a baseline wins, it gets published.
If a hybrid wins, it gets published.
If nothing beats the baselines, then nothing gets uploaded.

MagicQuant is here to:

* push quantization knowledge forward
* reveal model quirks
* challenge assumptions
* and give users the **strongest possible quantized versions** of any model that passes through the system

Whether that’s one model or twenty, the output will always reflect *quality, not quantity*.
