# Bad Trades, Early Pruning, and Search-Space Collapse

One of the most important parts of MagicQuant is what users never see in the final download table.

Before final survivors are selected, before nonlinear winners are found, and before the prediction engine is allowed to rank candidate hybrids, MagicQuant performs a brutal pruning pass over the search space.

This is where **bad trades** are removed.

A bad trade is not simply “a worse model.”

A bad trade is a candidate that offers too little practical size benefit for the damage it causes.

MagicQuant is built around the idea that a quantized model must earn its place. Smaller is not automatically better. Lower KLD is not automatically enough if the size cost is unreasonable. A candidate must be a meaningful trade.

---

## What “Bad Trade” Means

A bad trade usually means:

> **The model saved a small amount of space, but caused disproportionate damage.**

For example, from a real Qwen3-4B-Instruct-2507 MagicQuant run:

```text
Bad trade elimination:
'Q6_K' removed vs accepted anchor 'Q8_0' for 'embeddings'.

Reason:
small size gain (1.23%)
but disproportionate damage
(KLD x3.23, |PPL| x0.95)
````

In plain language:

- the size improvement was tiny
    
- the KLD damage was much worse
    
- the trade was not worth preserving
    

That is the heart of bad-trade pruning.

MagicQuant is not asking:

> “Is this technically smaller?”

It is asking:

> “Did this compression earn the damage it caused?”

If the answer is no, the candidate is removed.

---

## Why This Happens Before Prediction

MagicQuant has a prediction engine, but the predictor is not treated as magic.

The prediction engine is powerful because it is given a cleaner, more meaningful search space. It is not expected to rescue millions of obviously poor candidates.

Prediction becomes hardest when candidates are extremely close together, especially near measurement noise. If two quantizations differ by tiny amounts, the signal can become fuzzy. At that point, pretending the predictor can perfectly resolve every microscopic difference would be dishonest.

This is especially true in very low-bit regions.

Sub-4-bit quantization can be useful, but it can also become chaotic. Metrics can shift sharply, damage can accelerate, and tiny apparent differences may stop being meaningful.

So MagicQuant prunes first.

It removes candidates that are obviously bad, dominated, redundant, or too damaged before asking the predictor to reason about the remaining space.

---

## Isolation Probing: The First Reality Check

The pruning system begins with **isolation probes**.

An isolation probe starts from a high-precision carrier, usually BF16, and changes only one tensor group at a time.

Conceptually:

```text
Reference model:
  BF16 everywhere

Isolation sample:
  BF16 everywhere
  except attn_q = selected quant

Another isolation sample:
  BF16 everywhere
  except ffn_down = selected quant
```

This allows MagicQuant to measure the isolated effect of each tensor group.

The system can then ask:

- How much size did this group save?
    
- How much KLD damage did it cause?
    
- How much PPL delta did it cause?
    
- Is this group worth continuing?
    
- Which quant choices are obviously bad?
    
- Which choices are redundant?
    
- Which choices are safe anchors?
    

This is the foundation for everything that comes later.

---

## Real Example: Initial Probe Results

In one real Qwen3-4B-Instruct-2507 run, the earliest probe stage found the smallest baseline-candidate probes for several tensor groups:

```text
Isolation Group: attn_kv
Best savings: 3.73%
Winning candidate: IQ2_XXS
KLD=0.082588
PPLΔ=6.2856%

Isolation Group: attn_output
Best savings: 8.17%
Winning candidate: IQ2_XXS
KLD=0.074302
PPLΔ=8.1299%

Isolation Group: attn_q
Best savings: 8.17%
Winning candidate: IQ2_XXS
KLD=0.055446
PPLΔ=3.1076%

Isolation Group: embeddings
Best savings: 6.34%
Winning candidate: IQ2_XXS
KLD=0.005914
PPLΔ=0.4264%

Isolation Group: ffn_down
Best savings: 19.31%
Winning candidate: IQ2_XXS
KLD=0.162554
PPLΔ=13.7906%

Isolation Group: ffn_up_gate
Best savings: 38.80%
Winning candidate: IQ2_XXS
KLD=0.259223
PPLΔ=21.7417%
```

At first glance, some of these look attractive because the size savings are large.

But the damage tells the real story.

For example:

- `ffn_up_gate` at `IQ2_XXS` saved **38.80%**
    
- but produced **KLD 0.259223**
    
- and **PPLΔ 21.7417%**
    

That is not a subtle trade.

That is a warning flare.

MagicQuant uses these early probes to understand where compression becomes dangerous and where the trade stops being meaningful.

---

## Continuation: Not Everything Is Killed Immediately

Early probing does not automatically mean every low-bit result is removed immediately.

Some groups continue into deeper probing if the system sees that there is still meaningful structure to examine.

From the same run:

```text
Continuation enabled for 'embeddings':
smallest baseline-candidate probe 'IQ2_XXS' saved 6.34%.

Continuation enabled for 'attn_q':
smallest baseline-candidate probe 'IQ2_XXS' saved 8.17%.

Continuation enabled for 'attn_kv':
smallest baseline-candidate probe 'IQ2_XXS' saved 3.73%.

Continuation enabled for 'attn_output':
smallest baseline-candidate probe 'IQ2_XXS' saved 8.17%.

Continuation enabled for 'ffn_up_gate':
smallest baseline-candidate probe 'IQ2_XXS' saved 38.80%.

Continuation enabled for 'ffn_down':
smallest baseline-candidate probe 'IQ2_XXS' saved 19.31%.
```

This matters because MagicQuant is not blindly anti-low-bit.

It is willing to continue exploring when there may still be a useful trade.

But continuation does not mean survival.

It means:

> “There may still be information here worth measuring.”

---

## BF16 Suppression

Some groups show enough size-saving potential that keeping an explicit BF16 option stops being useful during the combination phase.

In the same run:

```text
BF16 tensor-choice suppressed groups: 2
  - ffn_up_gate
  - ffn_down
```

This happened because these groups showed large enough savings potential:

```text
ffn_up_gate: 38.80% best savings
ffn_down:    19.31% best savings
```

This does not mean BF16 is bad.

It means that for this search space, explicitly carrying BF16 as a tensor-choice option for those groups was no longer meaningful. The system had already determined that useful candidates existed below full precision, so BF16 could be suppressed to reduce pointless combinations.

---

## Search Space Before Final Optimization

After early probe analysis, the search space was still enormous.

The run had two active combination baselines:

```text
Active combo baselines: 2
  - Q8_0
  - UD-Q6_K_XL/Q6_K
```

For each base, there were many tensor-group choices.

Example under `Q8_0`:

```text
embeddings   => 11 choices
attn_q       => 19 choices
attn_kv      => 18 choices
attn_output  => 19 choices
ffn_up_gate  => 19 choices
ffn_down     => 19 choices
```

That produced:

```text
Base total: 25,803,558
```

And because there were two active bases:

```text
Grand total: 51,607,116
```

So before final pruning, MagicQuant was staring at:

> **51,607,116 possible combinations**

This is why pruning matters.

Without aggressive pruning, the system would waste time evaluating combinations that the early evidence already shows are unlikely to matter.

---

## Continuation Isolation Samples

Instead of brute-forcing those 51 million combinations, MagicQuant performs targeted continuation probing.

From the same run:

```text
Groups continuing after early probe: 6
Continuation isolation samples required: 108
Queued continuation samples: 108
```

Many of these were deduplicated or reused:

```text
Continuation sampling complete.
Completed: 36
Skipped existing: 72
Failed: 0
```

This is another important part of the system:

MagicQuant does not blindly rebuild what it already knows.

It reuses prior benchmark artifacts when possible and deduplicates equivalent isolation samples.

---

## Final Isolation Optimization

After continuation probing, MagicQuant performs final isolation optimization.

This is where the search space collapses.

Before:

```text
Grand total: 51,607,116
```

After:

```text
Active combo baselines: 1
  - Q8_0

Disabled combo baselines: 1
  - UD-Q6_K_XL/Q6_K
```

The surviving search space became:

```text
embeddings   => 1 choice
attn_q       => 4 choices
attn_kv      => 1 choice
attn_output  => 2 choices
ffn_up_gate  => 6 choices
ffn_down     => 10 choices
```

Final total:

```text
Base total: 480
Grand total: 480
```

That means MagicQuant reduced the search space from:

```text
51,607,116 combinations
```

to:

```text
480 combinations
```

That is a reduction of more than:

> **99.999%**

This is not a small optimization.

This is the system saying:

> “Almost all of this space is not worth asking the predictor about.”

---

## Why the External Carrier Was Disabled

The run also disabled one combination baseline:

```text
Disabled combination baseline 'UD-Q6_K_XL'
because all surviving carriers were below the meaningful uncovered-tensor threshold
and 'Q8_0' was selected as the deterministic safe representative.
```

In plain English:

MagicQuant checked whether the choice of base carrier meaningfully affected uncovered tensors.

It found that the remaining uncovered-tensor influence was below the configured meaningful threshold:

```text
meaningful threshold of 1.00%
```

So it collapsed the base carrier to the deterministic safe option:

```text
Q8_0
```

This avoided duplicating the entire combination space across carriers that no longer meaningfully differed.

That is another form of bad-trade thinking:

> If the carrier distinction no longer meaningfully changes the result, do not multiply the search space for it.

---

## Final Surviving Group Choices

After pruning, the group choices were dramatically smaller:

```text
embeddings:
  Q8_0

attn_q:
  IQ4_XS, Q5_K, Q6_K, Q8_0

attn_kv:
  Q8_0

attn_output:
  Q6_K, Q8_0

ffn_up_gate:
  IQ3_XS, IQ4_XS, UD-Q5_K_XL, Q6_K, UD-Q6_K_XL, Q8_0

ffn_down:
  UD-Q3_K_XL, IQ4_XS, IQ4_NL, Q4_K_S, UD-Q4_K_XL,
  Q5_K_S, Q5_K, Q6_K, UD-Q6_K_XL, Q8_0
```

This is the search space the predictor is finally allowed to reason over.

Not 51 million combinations.

Only 480.

The predictor is not being asked to find truth in a landfill.

It is being given the candidates that survived empirical sanity checks.

---

## Real Elimination Counts

The final pruning report showed:

```text
Learned-baseline eliminations: 0
Baselines skipped without learned rows: 0
Groups reduced to explicit-banned->Q8-fallback: 3
BF16-suppressed groups: 2
Hard damage eliminations: 11
Dominance eliminations: 12
Bad trade eliminations: 67
Disabled combination baselines: 1

Combination count before pruning: 51,607,116
Combination count after rule pruning: 480
```

These numbers matter.

They show that the final result is not produced by vibes.

The system removed:

- obviously damaged candidates
    
- dominated candidates
    
- redundant candidates
    
- bad trades
    
- meaningless carrier duplication
    

before prediction even began.

---

## Examples of Bad Trade Eliminations

### Embeddings

The embeddings group was especially strict.

Many candidates were removed against `Q8_0` because they saved very little size while causing much larger damage.

```text
'Q6_K' removed vs accepted anchor 'Q8_0' for 'embeddings'.

Reason:
small size gain (1.23%)
but disproportionate damage
(KLD x3.23, |PPL| x0.95)
```

The same reasoning removed multiple lower candidates:

```text
Q5_K
UD-Q5_K_XL
Q5_K_S
Q4_K_M
UD-Q4_K_XL
Q4_K_S
IQ4_NL
IQ4_XS
IQ3_S
UD-Q3_K_XL
IQ3_XS
```

In other words:

> For embeddings, the cheaper options were not good trades.

The group collapsed to:

```text
embeddings => Q8_0
```

This is exactly what MagicQuant should do when a group is too sensitive relative to the savings.

---

### Attention Q

For `attn_q`, some candidates survived, but many low-bit options were removed.

Final surviving options:

```text
attn_q => IQ4_XS, Q5_K, Q6_K, Q8_0
```

Examples of removed bad trades:

```text
'IQ3_S' removed vs accepted anchor 'IQ4_XS'
Reason:
small size gain (0.51%)
but disproportionate damage
(KLD x2.57, |PPL| x108.05)
```

```text
'IQ2_XXS' removed vs accepted anchor 'IQ4_XS'
Reason:
small size gain (1.38%)
but disproportionate damage
(KLD x17.25, |PPL| x1266.38)
```

This is a perfect example of why low-bit candidates are not automatically useful.

They may save a tiny amount more, but the damage can explode.

---

### Attention KV

For `attn_kv`, the system became even more conservative.

Final surviving option:

```text
attn_kv => Q8_0
```

Examples:

```text
'Q6_K' removed vs accepted anchor 'Q8_0'
Reason:
small size gain (0.58%)
but disproportionate damage
(KLD x1.77, |PPL| x117.30)
```

```text
'IQ4_XS' removed vs accepted anchor 'Q8_0'
Reason:
small size gain (1.09%)
but disproportionate damage
(KLD x8.29, |PPL| x71.09)
```

```text
'IQ2_S' removed vs accepted anchor 'Q8_0'
Reason:
small size gain (1.53%)
but disproportionate damage
(KLD x86.02, |PPL| x2559.37)
```

For this model, `attn_kv` did not offer a meaningful trade below `Q8_0`.

So MagicQuant stopped trying.

That is the point.

---

### Attention Output

For `attn_output`, the final options were:

```text
attn_output => Q6_K, Q8_0
```

Examples:

```text
'Q5_K' removed vs accepted anchor 'Q6_K'
Reason:
small size gain (0.66%)
but disproportionate damage
(KLD x1.97, |PPL| x55.64)
```

```text
'Q4_K_M' removed vs accepted anchor 'Q6_K'
Reason:
small size gain (1.28%)
but disproportionate damage
(KLD x5.09, |PPL| x104.14)
```

This shows that sometimes the useful cutoff is not Q8.

Here, `Q6_K` remained meaningful, but going below it stopped being worth the damage.

---

### FFN Up/Gate

This group produced huge savings, but also huge damage in the lowest-bit ranges.

Initial probe:

```text
IQ2_XXS:
savings=38.80%
KLD=0.259223
PPLΔ=21.7417%
```

Hard damage eliminations removed the worst options:

```text
IQ2_XXS
IQ2_XS
IQ2_S
IQ3_XXS
UD-IQ3_XXS
```

But the group still retained useful choices:

```text
ffn_up_gate =>
IQ3_XS, IQ4_XS, UD-Q5_K_XL, Q6_K, UD-Q6_K_XL, Q8_0
```

This is important.

MagicQuant did not say:

> “Low-bit is bad.”

It said:

> “These specific low-bit trades were too damaged, but this group still has meaningful lower-precision options worth considering.”

That is a more nuanced decision.

---

### FFN Down

For `ffn_down`, the final options were broader:

```text
ffn_down =>
UD-Q3_K_XL, IQ4_XS, IQ4_NL, Q4_K_S, UD-Q4_K_XL,
Q5_K_S, Q5_K, Q6_K, UD-Q6_K_XL, Q8_0
```

But the worst candidates were still removed:

```text
Hard damage elimination:
'IQ2_XXS' removed
savings=19.31%
KLD=0.162554
PPLΔ=13.7906%
```

```text
Hard damage elimination:
'IQ2_XS' removed
savings=19.00%
KLD=0.108203
PPLΔ=7.8415%
```

This group had a wider usable range, but still had a damage floor.

---

## Why No Sub-4-Bit Hybrids Survived

In this run, MagicQuant did explore below 4-bit territory.

Some groups had low-bit candidates available during probing, and one group retained a 3-bit candidate after pruning.

But no sub-4-bit hybrid survived as a final meaningful output.

Why?

Because those candidates failed to become competitive under:

- dominance checks
    
- bad-trade checks
    
- nonlinear trade evaluation
    
- final validation
    

At some point, MagicQuant effectively says:

> “This region exists, but I cannot find a meaningful hybrid here under the current rules.”

That does not mean sub-4-bit baselines are useless.

It means:

> MagicQuant did not find a hybrid in that region that was worth publishing.

So the final output may still include sub-4-bit baselines from llama.cpp or external providers, but MagicQuant will not fabricate a hybrid just to fill the space.

---

## When MagicQuant “Gives Up” on a Region

MagicQuant is allowed to stop fighting in a region.

This is intentional.

If the system reaches a point where:

- damage is too high
    
- measurement noise overwhelms meaningful differences
    
- candidates are too close to distinguish confidently
    
- or the trade curve does not produce useful nonlinear winners
    

then MagicQuant stops trying to force a result.

The philosophy is:

> If MagicQuant cannot measure a meaningful improvement, it should not pretend one exists.

This is why some final releases may have no MagicQuant hybrid in certain ranges.

That is not a failure.

That is restraint.

---

## Why This Builds Trust

The final survivor table only shows what lived.

But the pruning logs show what died and why.

That matters because it proves MagicQuant is not simply choosing a few attractive models after the fact.

It is enforcing rules before prediction:

- hard damage rules
    
- dominance rules
    
- bad-trade rules
    
- equivalent-truth rules
    
- carrier-collapse rules
    
- group-level pruning rules
    

The final output is smaller because the system is deliberately hostile to weak evidence.

---

## Practical Meaning

When MagicQuant reduces:

```text
51,607,116 combinations
```

to:

```text
480 combinations
```

it is not saying:

> “There is absolutely no possible interesting combination outside this space.”

It is saying:

> “Given the measured isolation evidence, the removed space is so unlikely to matter that it is not worth spending benchmark time on.”

That distinction is important.

MagicQuant is not claiming omniscience.

It is claiming practical judgment.

The system is designed to find candidates worth validating, not to exhaust every theoretical possibility.

---

## Summary

A bad trade is a candidate that does not earn its cost.

MagicQuant removes bad trades before prediction because:

- they distort the search space
    
- they waste benchmark time
    
- they make prediction harder
    
- and they are unlikely to become meaningful final survivors
    

In the Qwen3-4B-Instruct-2507 example, the system reduced the search from:

> **51,607,116 combinations**

to:

> **480 combinations**

before final prediction.

That collapse happened through:

- isolation probing
    
- continuation sampling
    
- hard damage elimination
    
- dominance elimination
    
- equivalent-truth elimination
    
- bad-trade elimination
    
- BF16 suppression
    
- carrier collapse
    

The result is a cleaner, smaller, more meaningful candidate space.

MagicQuant does not try to find every possible hybrid.

It tries to find the hybrids that deserve to exist.

---

## Core Principle

> **A smaller model is not automatically a better model.  
> A hybrid is not automatically a useful hybrid.  
> A trade only matters if it earns the damage it causes.**

