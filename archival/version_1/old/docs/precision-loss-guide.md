# **Precision Loss: A Practical Philosophy for MagicQuant**

## **Overview**

Quantization is a powerful way to shrink and accelerate large language models, but it always comes at a cost. That cost is **precision loss**: a measurable drift between how a quantized model behaves compared to its original BF16/F16 version.

> "Precision Loss" is the referred statement for the the perplexity drift % (aka: PPL Delta percentage) which utilizes the llama.cpp tool for measurement.

**MagicQuant rejects any quantization with more than 5% precision loss.**
This does *not* mean >5% loss is useless. What it means is: it no longer meets the quality guarantees this project exists to provide.

MagicQuant isn’t about “making toys.” It’s about producing **small, fast models that preserve the original model’s intelligence**, especially for trustworthy or agentic automation.

---

# **What Is Precision Loss?**

Quantization does not remove knowledge; it compresses it. But in compression, weights shift. When enough weights shift, the model’s behavior drifts from the original.

Precision loss captures that drift numerically.

* **0% precision loss** → output is identical to the base model
* **higher precision loss** → more drift, more instability, more hallucination

A “better” answer from a quantized model is *not* evidence that it improved. It's almost always an **accidental correct hallucination**, not a fidelity gain.

The goal is not to make the model *different*. The goal is to keep it **as close as possible** to the BF16/F16 original.

---

# **Precision Loss Tiers (MagicQuant Philosophy)**

You will see quantization discussed in terms like Q8, Q6_K, Q5, Q4.
Those labels are not enough. They’re vague, inconsistent, and misleading—because different architectures react differently to the same quant.

MagicQuant evaluates quantization by **precision loss percentage**, not by bit-width mythologies.

Below is the practical breakdown.

---

## **0.0% – 0.1%

“Scientific Exactness” | Functionally Identical**

This is the **god-tier** zone.

* Indistinguishable from BF16/F16
* Used when you need scientific reproducibility or ultra-deterministic agent behavior
* Necessary only for very tiny models (350M–1B) where micro-drift has outsized impact
* Overkill for most real applications, but still beautiful when achievable

This level of precision loss is so tiny it often exceeds the precision of your evaluation methods themselves.

---

## **0.1% – 1%

“Near-Lossless” | Production-Grade Fidelity**

This is where *serious* work happens.

Models in this range retain **99–99.9%** of their “brain.”
They handle:

* complex reasoning
* long-form tasks
* multi-step agentic workflows
* repeatable automation
* semantic fidelity across thousands of queries

If you’re doing agentic work, *especially* agentic work, this is your sweet spot.

> For automation that must run cleanly for thousands of cycles,
> **0.1% to 1% precision loss is the only safe range.**

---

## **1% – 3%**

“Minimal Loss” | High-Quality Personal Use**

Once you cross 1%, you’re no longer in near-lossless territory.
You’re entering the “minimal loss” range:

* still solid
* still coherent
* rarely derails
* great for personal chatbot usage
* still acceptable for many dev workflows

But subtle degradations begin appearing:

* coherence cracks
* semantic drift
* rare-but-annoying hallucinations
* degradation during long chains of reasoning
* higher instability in agent loops

**MagicCodingMan's personal limit:** *2–2.5% preferred, 3% absolute maximum.*
This is where quality and stability start slipping perceptibly, even if still usable.

---

## **3% – 5%

“Borderline” | Usable But Noticeably Weaker**

This is the upper limit allowed by MagicQuant.

Why?

Because past this point:

* hallucination increases sharply
* logical consistency decays
* long-context coherence becomes unreliable
* small drift becomes visible drift
* agentic stability drops off a cliff

This range might still be fine for:

* tinkering
* casual chatting
* personal local LLM fun

…but it no longer aligns fully with the quality expectations of MagicQuant.

---

# **5%+ Precision Loss

“Toy Zone” | Outside MagicQuant’s Philosophy**

Some people in the community will tell you:

> “Oh yeah, Q3 or 10–20% precision loss still works fine for chatting!”

And sure.
If your model’s job is to entertain you and occasionally say something funny, then maybe it’s “fine.”

But in MagicQuant’s philosophy:

* **Past 5% = noticeable brain damage**
* **Past 10% = major brain damage**
* **Q3 = toy territory**
* **Usable for curiosity, not reliability**

MagicQuant does *not* support, optimize, or endorse:

* Q3
* double-digit precision loss
* “it kinda works if you squint” quantizations

That’s not the purpose of this project.

---

# **Why MagicQuant Uses a 5% Hard Limit**

MagicQuant exists to find the **smallest, fastest models that remain trustworthy**.

That means:

* agentic consistency
* semantic fidelity
* repeatable automation
* reproducible reasoning
* minimal hallucination
* stability under long inference sessions

If a quantization drifts more than 5% from the base model, these qualities collapse.

So the project draws a line:

> **If it’s above 5% loss, it doesn’t belong in MagicQuant.
> Not because it’s useless, because it’s untrustworthy.**

---

# **The Reality: “Q8 > Q6” Is Wrong 30–40% of the Time**

People love talking about quants in terms of “Q8 = best, Q6 = slightly worse, Q4 = lower, Q3 = junk.”

That myth holds true…
**about 60–70% of the time.**

MagicQuant’s data shows:

* Some models prefer Q6 over Q8
* Some prefer hybrid patterns
* Some quantization schemes damage FFNs more than self-attention blocks
* Some MoE models survive Q4 where dense models die
* Some weight groups tolerate lower precision far better than others

This is why MagicQuant evaluates all quantizations based on:

**precision loss percentage, not quant scheme**
**real-world benchmark drift, not bit-width hype**
**actual reasoning fidelity, not theoretical assumptions**

This is the heart of the project.

---

# **Final Philosophy Summary**

* **0–0.1%** → God-tier, scientifically exact
* **0.1–1%** → True near-lossless, agent-ready
* **1–3%** → Minimal loss, great for personal use
* **3–5%** → Borderline, but still functional
* **5%+** → Toys, not tools, outside MagicQuant’s scope

MagicQuant is fundamentally about **trustworthy downsizing**.
If a downsized model cannot retain fidelity, it doesn’t matter that it’s small.
It matters that it’s no longer the model.

---

# **FAQ: Precision Loss, Model Size, and MagicQuant Philosophy**

## **If I can’t fit a low-precision-loss quant on my GPU, should I switch to a smaller model instead?**

**Yes. Nearly 99.99% of the time, yes.**

If your GPU can only fit a large model by using something like **Q3 (10–20% precision loss)**, that model is already deep in “toy” territory. Nearly any smaller model with *good* precision loss will outperform it in coherence, reliability, and reasoning.

Examples:

* If a **20B** only fits at *10% precision loss*, but a **14B** fits at **Q4 (~3–5%)**, choose the 14B.
* If an **14B** fits at **Q6_K (~1–3%)**, and you want a reliable chat bot, this would be a great pick!
* If a **8B** fits at **Q8 (<1% loss)** and if you need high-fidelity agentic work, this might even be the best pick.

This also assumes models perform “linearly” with size (which hint hint, they do not).
Some 4B models hit *way above their weight class* for example.

**Quality beats quantity every time.**

---

## **Why do you call 5%+ precision loss an “experimental toy”?**

It’s not an insult and I’m not judging anyone’s preferences.
It’s dramatic phrasing for a simple truth:

> If I cannot trust the model to respond accurately, maintain coherence, or avoid hallucinations,
> then I personally cannot treat it as a serious tool.

Past ~5% loss, drift compounds, reliability decreases, and reasoning consistency cracks.
For **exploration, tinkering, or fun**, that’s fine.
For **real work**? It’s no longer trustworthy.

MagicQuant exists to produce **trustworthy, dependable quants**, so anything over 5% sits outside that mission.

---

## **If you personally prefer <3% loss, why does MagicQuant allow models up to 5% loss?**

Because not everyone has the same needs or the same tolerance.

I’m picky. Painfully picky.
I can feel the degradation at 3% the same way an audio engineer hears a bad bitrate.

But MagicQuant isn’t about *my* personal threshold.
It’s about providing **high-quality models for a wide range of users**.

So the limit is:

* **3%** → my personal “maximum for serious use”
* **5%** → the project’s “maximum acceptable loss for public release”

Anything above that no longer meets the reliability standards that MagicQuant promises.

---

## **Why is MagicQuant so strict about precision loss anyway?**

Because **trust is everything**.

MagicQuant’s goal is to ensure:

* every release is benchmarked
* every quant is justified
* no model is included “just because it fits”
* Baselines are only included if they prove they're worthy to exist
* hybrids aren’t included unless they outperform standard quants
* every upload meets a strict fidelity standard

Other repos often upload Q8, Q6, Q5, Q4 simply because that’s “what people expect.”

MagicQuant uploads a quant **only** when the numbers prove it deserves to exist.

You can always have faith that:

* the quant represents the best possible tradeoff
* precision loss is within acceptable bounds
* drift is measured, not guessed
* you’re getting a *known-good* version
* nothing is half-tested or blindly exported

This is why MagicQuant’s benchmark sheets and testing pipeline exist.

---

## **Why didn’t you cover the “1% to 2%” precision loss range in detail? Is that range bad?**

Not bad, just *awkward*.

Here’s why:

### 0.1% – 1%

This is the **near-lossless** zone.
Perfect for agentic workflows, tools, reasoning, long chains, etc.

### 1% – 2%

This zone is still excellent, but…

Most users fall into two buckets:

1. **They need near-lossless fidelity**
   → so they aim for <1%

2. **They want the smallest model possible**
   → so they’re fine going to 2–5% loss for the size/TPS gains

That leaves the 1–2% tier in a weird middle-ground that doesn’t have strong demand.

It’s not that this range is *bad*.
It’s just rarely the optimal choice for most people’s goals.
