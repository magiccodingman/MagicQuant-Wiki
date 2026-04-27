# MagicQuant Prediction Engine

## Isolation-Based Rank-Safe Damage Estimation

MagicQuant is not trying to be omniscient.

It is trying to do something more useful: take an impossibly large quantization search space, learn the physical behavior of that model from a small set of real isolated measurements, predict where the meaningful trade spaces are likely to be, and spend expensive full benchmarks only where the prediction has a real chance of changing the final downloadable model set.

That distinction matters.

MagicQuant does not claim:

> “This predicted KLD is always exactly right.”

MagicQuant claims:

> “This prediction is good enough to decide where reality is worth checking.”

And then MagicQuant checks.

If the real benchmark fails the required outcome, the candidate is rejected. It does not survive because it was close. It does not survive because the prediction was elegant. It does not survive because the math liked it. The real benchmark owns the final truth.

The prediction engine is therefore best understood as a **search-space compression engine**, not a final authority. It converts trillions, billions, or millions of possible tensor-group quantization combinations into a much smaller list of candidates that are worth physically building and benchmarking.

That is the heart of MagicQuant:

> Predict aggressively.  
> Validate brutally.  
> Publish only survivors.

---

# 1. The Core Philosophy: Do Not Fight for Pennies

The modern MagicQuant pipeline was built around a hard-earned observation:

Most of the quantization search space is not worth fighting over.

That sounds almost heretical at first, because hybrid quantization is full of deliciously weird nonlinear behavior. Sometimes a lower-bit quant wins in a specific tensor group. Sometimes two groups interact in a way that makes a combination better than expected. Sometimes a smaller quantization choice fits a tensor group’s distribution better than a larger one.

Those effects are real.

But most of the time, they are either:

1. predictable from isolated samples,
    
2. too small to matter,
    
3. within measurement noise,
    
4. not large enough to change the final Pareto frontier,
    
5. or rejected by real validation anyway.
    

MagicQuant’s philosophy is not “nonlinear behavior does not exist.”

MagicQuant’s philosophy is:

> Nonlinear behavior exists, but not all nonlinear behavior deserves combinatoric tribute.

That is the “do not fight for pennies” rule.

A system can waste enormous time chasing a tiny local inversion that technically exists but does not produce a meaningful downloadable model. MagicQuant instead tries to identify the combinations that are likely to be meaningful: lower KLD at the same or smaller size, better-than-linear interior tradeoffs, or genuinely useful size/fidelity gaps between anchors.

The prediction engine exists to find those opportunities.

The validation system exists to make sure they are real.

---

# 2. The Monotonic Degradation Assumption

## 2.1 What MDA Means

The **Monotonic Degradation Assumption**, or **MDA**, is the foundation of MagicQuant’s prediction engine.

The name sounds fancy, but the core idea is simple:

> If every part of a model is held constant except one tensor group, and that tensor group is changed from a higher-fidelity quantization to a lower-fidelity quantization, the lower-fidelity version is expected to have equal or worse KLD most of the time.

For example, imagine a model has seven active tensor groups.

You randomly choose a complete hybrid configuration:

```text
embeddings  = Q8_0
attn_q      = IQ4_XS
attn_kv     = Q6_K
attn_output = Q8_0
ffn_up_gate = IQ4_XS
ffn_down    = Q6_K
lm_head     = native/exact
```

Now freeze six of those groups.

Only one group is allowed to change.

For example:

```text
attn_output = Q8_0
```

versus:

```text
attn_output = Q6_K
```

The MDA says that, in most contexts, the Q8_0 version should have equal or lower KLD than the Q6_K version.

Not always.

Most.

That “most” is where the whole philosophy lives.

## 2.2 MDA Is Not a Hard Law

MDA is not a universal law of physics.

It is an empirical operating assumption.

There are real exceptions. Sometimes Q6_K can beat Q8_0 in a specific tensor group under a specific surrounding combination. Sometimes a lower-bit quantization produces a distribution that better fits a tensor group. Sometimes KLD shifts in tiny ways that are measurable but not practically decisive.

MagicQuant does not deny those cases.

It classifies them.

The important distinction is:

```text
Unexpected does not automatically mean important.
```

A lower-bit quant beating a higher-bit quant by `0.000001` KLD may be mathematically interesting, but it may not matter to the final model list. If that difference is inside measurement noise, or if it does not change survivor ranking, or if the resulting file does not earn a meaningful size/fidelity slot, MagicQuant should not burn the universe to chase it.

That is the practical meaning of MDA.

## 2.3 The Observed Pattern

Across large-scale MagicQuant sampling, the monotonic pattern is extremely strong.

In your phrasing:

> If six tensor groups are locked and the seventh group compares Q8_0 against Q6_K, Q8_0 beats Q6_K about 98.4% of the time.

That is the key empirical insight.

The remaining ~1.6% is where things get interesting. But most of that unexpected region tends to fall inside noise or near-noise. A much smaller fraction contains measurable inversions where the lower-bit option genuinely wins.

Those inversions matter as evidence that nonlinear quantization behavior exists.

They do not automatically matter enough to dominate the final pipeline.

The prediction engine is designed around that reality.

---

# 3. What MagicQuant Is Actually Predicting

MagicQuant predicts whether a hybrid quantized model is likely to land in a meaningful region of the size/fidelity frontier.

The main predicted values are:

```text
predicted size
predicted KLD
predicted PPL
predicted rank / ordering
predicted gain over a local tradeoff line
```

But KLD is the star of the show.

## 3.1 KLD as Damage

In MagicQuant documentation, KLD can be thought of as **damage**.

Lower KLD means the quantized model behaves closer to the reference behavior under the benchmark distribution.

Higher KLD means more divergence.

So when the docs say “damage,” they generally mean:

```text
damage ≈ KLD introduced by quantization
```

This does not mean KLD captures every kind of quality loss. It does not mean KLD perfectly predicts subjective model feel. It does not mean KLD replaces human testing.

But KLD is highly useful because it gives MagicQuant a stable numerical signal for comparing quantization choices under controlled benchmark conditions.

## 3.2 The Goal Is Not Merely Low KLD

A model with lower KLD is not automatically a better final output.

Size matters too.

A 4.0 GB model with KLD `0.0013` and a 3.99 GB model with KLD `0.00131` are probably not meaningfully different to users. Likewise, a 2.41 GB hybrid that is slightly worse than a baseline at the same size does not deserve to survive just because it is fancy.

MagicQuant cares about the frontier:

```text
For the size paid, did the model earn the damage level it achieved?
```

This is why the pipeline compares candidates against:

1. strict dominance,
    
2. nearby baseline replacement,
    
3. interior better-than-linear tradeoff opportunities,
    
4. final dominance,
    
5. and meaningful spacing.
    

The prediction engine is only useful because it feeds into this stricter survival system.

---

# 4. The Isolation Sampling Strategy

MagicQuant does not try to understand the whole combinatoric space directly.

That would be brutal.

Instead, it learns how each tensor group behaves when isolated.

## 4.1 Tensor Groups

A model is divided into major tensor groups such as:

```text
embeddings
lm_head
attn_q
attn_kv
attn_output
ffn_up_gate
ffn_down
moe_experts
moe_router
```

Not every architecture uses every group. Unused groups are ignored for that model.

A hybrid configuration can be thought of as:

```text
base quant + optional group-specific overrides
```

For example:

```text
base        = Q8_0
embeddings  = Q8_0
attn_kv     = Q8_0
attn_output = Q6_K
attn_q      = IQ4_XS
ffn_up_gate = IQ4_XS
ffn_down    = IQ4_NL
```

If a group has no explicit override, it inherits the base quant.

## 4.2 Effective Quantization

For every tensor group `g`, MagicQuant determines an **effective baseline**:

```text
effective(g, combo) =
    group override if one exists
    otherwise combo base quant
```

So if the base quant is Q6_K and `attn_output` has no override, then:

```text
effective(attn_output) = Q6_K
```

If the base quant is Q6_K but `attn_output = Q8_0`, then:

```text
effective(attn_output) = Q8_0
```

This matters because prediction is based on the final effective quantization state of each group, not just the written override list.

## 4.3 The Q8 Carrier

MagicQuant uses a Q8 carrier isolation strategy.

The idea is:

1. Build a Q8-based model.
    
2. Hold every active tensor group at native/exact precision, such as BF16/F16/native exact depending on the run.
    
3. Change exactly one tensor group to the target quantization.
    
4. Benchmark the result.
    
5. Store the measured KLD and size behavior for that group/quant pair.
    

This creates a single-group isolation measurement:

```text
isolation_damage(group, quant)
```

For example:

```text
isolation_damage(attn_output, Q6_K)
isolation_damage(attn_output, Q8_0)
isolation_damage(ffn_down, IQ4_XS)
isolation_damage(attn_kv, Q5_K)
```

These measurements become the physical data layer of the prediction engine.

## 4.4 Why Isolation Works

Isolation works because tensor-group damage is often approximately additive.

If damaging `attn_q` by itself causes some amount of KLD shift, and damaging `ffn_down` by itself causes another amount, then damaging both together often lands near the sum of those effects.

Not perfectly.

But often enough to be powerful.

That is the crack in the combinatoric wall.

MagicQuant exploits that crack.

Instead of benchmarking every possible hybrid combination, it benchmarks a carefully chosen set of isolated tensor-group behaviors and uses those measurements to predict large regions of the combination space.

---

# 5. The Three-Layer Prediction Stack

The old validator names these layers M1, M4, and M5.

This documentation uses clearer names:

|Old validator name|Documentation name|Purpose|
|---|---|---|
|M1|**Isolation Additive Backbone**|Stable rank backbone based on summed isolated damage|
|M4|**Bit-Stress Interaction Estimator**|Numeric correction for low-bit cross-group compounding|
|M5|**Rank-Safe Projection**|Final monotone projection that keeps the rank discipline of the additive backbone while preserving as much interaction accuracy as possible|

The important pair is:

```text
Isolation Additive Backbone + Rank-Safe Projection
```

The first gives MagicQuant a stable physical ordering.

The second prevents the interaction model from making locally chaotic ranking claims.

Together, they form the practical prediction engine.

---

# 6. Layer One: Isolation Additive Backbone

## 6.1 Purpose

The **Isolation Additive Backbone** is the simplest and most important estimator.

It asks:

> If every tensor group contributes its isolated KLD damage independently, what is the total predicted damage of this hybrid?

This is the rank spine of the prediction engine.

It is not always the most numerically accurate layer, but it is usually extremely good at ordering combinations.

## 6.2 Definitions

Let:

```text
G = set of active tensor groups
c = a candidate hybrid configuration
e_g(c) = effective quantization for group g in candidate c
D(g, q) = isolated measured KLD damage for group g at quant q
```

Some effective states are treated as zero-damage aliases:

```text
Q8_0
native exact
BF16 exact
F16 exact
```

In the current prediction model, Q8_0 is treated as the practical carrier floor for marginal damage. Native/exact aliases are also zero because they are not contributing quantization damage in the isolated sense.

So:

```text
D(g, Q8_0) = 0
D(g, native/exact) = 0
D(g, BF16/F16 exact) = 0
```

For all other quantizations, MagicQuant looks up the isolated measurement:

```text
D(g, q) = measured KLD from the Q8-carrier single-group isolation
```

## 6.3 Formula

The additive prediction is:

```text
A(c) = Σ D(g, e_g(c))
       over all active tensor groups g
```

Expanded:

```text
predicted_additive_kld(candidate)
    =
    damage(embeddings,  effective_embeddings)
  + damage(lm_head,     effective_lm_head)
  + damage(attn_q,      effective_attn_q)
  + damage(attn_kv,     effective_attn_kv)
  + damage(attn_output, effective_attn_output)
  + damage(ffn_up_gate, effective_ffn_up_gate)
  + damage(ffn_down,    effective_ffn_down)
  + damage(moe_experts, effective_moe_experts)
  + damage(moe_router,  effective_moe_router)
```

Groups not present in the architecture are ignored.

Groups at Q8/native/exact contribute zero.

Groups with missing isolation data make the candidate unsafe or incomplete for prediction.

## 6.4 Why This Works

The additive model works because most quantization damage behaves approximately monotonically and approximately independently at the tensor-group level.

It is not perfect.

But it captures a huge amount of the useful signal.

The additive backbone is especially important for ranking because it is conservative. It does not invent complex interaction stories. It says:

> “Based on isolated physical measurements, this combination should be less damaged than that one.”

That is a powerful baseline.

## 6.5 The Role of MDA

The Isolation Additive Backbone is where the MDA becomes operational.

If Q8_0 isolated damage is lower than Q6_K isolated damage for `attn_output`, then the additive backbone generally assumes Q8_0 is the safer choice for `attn_output` across combinations.

That assumption can miss nonlinear inversions.

But the data says it is overwhelmingly right in practical ranking terms.

And when it is wrong, MagicQuant’s validation layer catches the important cases that are actually tested.

---

# 7. Layer Two: Bit-Stress Interaction Estimator

## 7.1 Purpose

The additive backbone is stable, but it is intentionally simple.

Quantization damage is not always perfectly additive. Low-bit choices can interact. Two mildly damaged groups may combine into more damage than either isolation measurement suggests.

The **Bit-Stress Interaction Estimator** adds a learned correction term for this.

It asks:

> When multiple groups are pushed into low-bit territory together, how much extra interaction damage should be expected?

## 7.2 Bit Stress

Each quantization has an approximate bit range.

For example:

```text
Q8_0  ≈ 8-bit
Q6_K  ≈ 6-bit
Q5_K  ≈ 5-bit
Q4_K  ≈ 4-bit
IQ4_* ≈ 4-bit-ish
IQ3_* ≈ 3-bit-ish
```

MagicQuant chooses a candidate bit-stress threshold `B`.

For each group, stress is:

```text
stress(g) = max(0, B - bits(e_g))
```

So if `B = 8`:

```text
Q8_0  → stress 0
Q6_K  → stress 2
Q5_K  → stress 3
Q4_K  → stress 4
IQ3_* → stress 5
```

The lower the bit depth, the higher the stress.

## 7.3 Pairwise Cross-Term

For every pair of active damaged groups, MagicQuant computes a pairwise interaction contribution:

```text
cross(g, h) =
    D(g, e_g)
  × D(h, e_h)
  × stress(g)
  × stress(h)
```

Then sums across all pairs:

```text
X_B(c) = Σ [D(g, e_g) × D(h, e_h) × stress(g) × stress(h)]
         for all g < h
```

This term is zero when groups are not stressed.

It grows when multiple low-bit damaged groups are active together.

## 7.4 Interaction Formula

The interaction prediction is:

```text
Y(c) = α × A(c) + β × X_B(c)
```

Where:

```text
A(c) = additive backbone prediction
X_B(c) = bit-stress pairwise cross-term
α = fitted additive scale
β = fitted interaction scale
B = selected bit-stress threshold
```

MagicQuant fits `α`, `β`, and the best threshold candidate against existing benchmark truth for the active model/imatrix bucket.

This matters: the fit is scoped.

MagicQuant is not blindly applying one universal correction across every architecture, model, and imatrix. It fits within the current prediction context when enough benchmark rows exist.

## 7.5 Why This Layer Exists

The interaction estimator improves numeric accuracy when the additive backbone underestimates compounding damage.

This is especially useful in lower-bit regions, where quantization gets spicy.

Sub-4-bit territory is where things can get especially weird. Damage can stop behaving politely. Measurement noise can become more noticeable. Tensor groups that looked tame in isolation can interact in stranger ways.

The interaction estimator gives MagicQuant a way to bend the prediction toward measured reality without throwing away the additive backbone.

But it still has a weakness:

> A fitted interaction model can become locally too confident.

That is why the final projection layer exists.

---

# 8. Layer Three: Rank-Safe Projection

## 8.1 Purpose

The **Rank-Safe Projection** is the reconciliation layer.

It combines:

```text
the stable ordering of the Isolation Additive Backbone
```

with:

```text
the improved numeric estimate of the Bit-Stress Interaction Estimator
```

It does this by forcing the final prediction to remain monotonic in additive-backbone order.

In plain English:

> If the additive isolation evidence says candidate A should be no worse than candidate B, the final projected prediction is not allowed to say B is clearly better than A unless they collapse into a tied plateau.

That is the stabilizer.

## 8.2 Sort Order

MagicQuant sorts predictable candidates by:

```text
additive predicted KLD
then interaction predicted KLD
then predicted size
```

The Python validator version sorts by additive prediction and interaction prediction. The C# implementation also includes predicted size as an additional tie-breaker.

The primary key is the additive backbone.

That means the final prediction treats additive isolation order as the rank spine.

## 8.3 Optimization Problem

Let:

```text
A_i = additive prediction for candidate i
Y_i = interaction prediction for candidate i
Z_i = final rank-safe projected prediction
```

Sort candidates so that:

```text
A_1 ≤ A_2 ≤ A_3 ≤ ... ≤ A_n
```

The Rank-Safe Projection solves:

```text
minimize Σ (Z_i - Y_i)^2

subject to:

Z_1 ≤ Z_2 ≤ Z_3 ≤ ... ≤ Z_n
```

In words:

> Find the final prediction sequence that stays as close as possible to the interaction estimator while never violating the additive backbone order.

This is classical isotonic regression.

MagicQuant computes it with the **Pool Adjacent Violators Algorithm**, or PAVA.

## 8.4 What PAVA Does

PAVA walks through the interaction predictions in additive order.

If the interaction predictions are already monotonic, nothing changes.

Example:

```text
additive order:      A1   A2   A3   A4
interaction values:  .01  .02  .03  .04
projected values:    .01  .02  .03  .04
```

No problem.

But if the interaction model violates additive order:

```text
additive order:      A1   A2   A3   A4
interaction values:  .01  .04  .03  .05
```

Then `.04` before `.03` is a violation.

PAVA pools the violating adjacent values:

```text
(.04 + .03) / 2 = .035
```

Final:

```text
projected values:    .01  .035 .035 .05
```

The disputed region becomes a plateau.

## 8.5 What Plateaus Mean

A plateau is not a bug.

A plateau is MagicQuant saying:

> “The estimators disagree here, and the honest answer is that this region should be treated as tied or uncertain.”

That is much better than pretending to know the exact ordering of candidates separated by microscopic noisy differences.

A plateau means:

```text
The interaction estimator wanted to locally reorder candidates.
The additive backbone did not support that reorder.
The projection pooled the disputed values.
```

This is a principled uncertainty signal.

## 8.6 Why This Is Powerful

The Rank-Safe Projection has the perfect personality for this problem:

1. It does not introduce new tensor-group assumptions.
    
2. It does not require a new fitted model.
    
3. It preserves the practical monotonic structure learned from isolated samples.
    
4. It keeps the interaction estimator’s numerical improvements wherever they do not violate rank safety.
    
5. It converts disagreement into plateaus instead of fake precision.
    

This is why the additive backbone and rank-safe projection stabilize each other.

The additive backbone is the skeleton.

The interaction estimator is the muscle.

The rank-safe projection is the nervous system saying:

> “Great, but do not flail.”

---

# 9. Full Prediction Formula

For a candidate hybrid `c`:

## 9.1 Effective Baselines

```text
e_g(c) =
    override_g(c), if group g has an explicit override
    base(c), otherwise
```

## 9.2 Isolated Damage Lookup

```text
D(g, q) =
    0, if q is Q8_0 or native/exact
    measured isolated KLD for group g at quant q, otherwise
```

## 9.3 Additive Backbone

```text
A(c) = Σ D(g, e_g(c))
```

## 9.4 Bit Stress

```text
stress_B(g, c) = max(0, B - bits(e_g(c)))
```

## 9.5 Pairwise Cross-Term

```text
X_B(c) =
    Σ D(g, e_g(c)) × D(h, e_h(c)) × stress_B(g, c) × stress_B(h, c)
    for all g < h
```

## 9.6 Interaction Estimate

```text
Y(c) = max(0, α × A(c) + β × X_B(c))
```

## 9.7 Rank-Safe Projection

Sort all candidates by:

```text
A(c), then Y(c), then predicted size
```

Then solve:

```text
Z* = argmin Σ (Z(c) - Y(c))²

subject to:

Z(c_1) ≤ Z(c_2) ≤ ... ≤ Z(c_n)
```

The final predicted KLD is:

```text
PredictedKld(c) = Z*(c)
```

That final value is what MagicQuant uses for rank-safe candidate selection.

---

# 10. Size Prediction

KLD is only half the story.

MagicQuant also predicts size.

The size estimator follows a similar isolation idea.

It starts from a base-only exact blanket size anchor, then adds isolated size deltas for each effective group quantization.

Conceptually:

```text
PredictedSize(c)
    =
    base_only_size(base(c))
  + Σ [isolated_size(g, e_g(c)) - q8_exact_blanket_size]
```

The reason for subtracting the Q8 exact blanket size is that each group isolation snapshot includes the common carrier structure. MagicQuant wants only the marginal size delta for the group override.

If size prediction is not safe because required isolation size anchors are missing, the candidate can be marked unsafe for selection.

This matters because MagicQuant’s final decision is not:

```text
Did predicted KLD look good?
```

It is:

```text
Did predicted KLD look good at the predicted size, and then did the real built model confirm that outcome?
```

---

# 11. Prediction Is Not Validation

This is one of the most important parts of the system.

MagicQuant does not publish predictions.

MagicQuant publishes benchmark-validated survivors.

A prediction can be impressive, elegant, and almost correct — and still be rejected.

The final question is not:

```text
Was the predicted KLD within X error?
```

The final question is:

```text
Did the candidate achieve the outcome it was selected for?
```

That is a much better test.

## 11.1 Strict Dominance Replacement

A candidate selected for strict dominance is trying to replace an existing anchor.

It must prove:

```text
actual_size(candidate) ≤ actual_size(anchor)
actual_KLD(candidate)  < actual_KLD(anchor)
```

With the configured epsilon applied to avoid meaningless equality games.

If it fails, it is rejected.

Even if the prediction was close.

## 11.2 Near-Baseline Replacement

A near-baseline candidate is allowed to sit within a small size growth window near an existing anchor.

But it must beat the local linear KLD expectation between anchors.

It is not enough to be “kind of nearby.”

It has to earn the slot.

## 11.3 Interior Subspace Discovery

Interior discovery looks between adjacent surviving anchors.

Suppose there is a smaller/higher-damage anchor and a larger/lower-damage anchor.

MagicQuant draws a local linear KLD line between them:

```text
line_KLD(size)
```

A candidate inside that window must land below that line:

```text
actual_KLD(candidate) < line_KLD(actual_size(candidate))
```

That means it is not merely between the two models.

It is better than the expected straight-line tradeoff.

That is what makes it interesting.

## 11.4 Final Dominance and Spacing

Even after candidates validate, MagicQuant still performs final cleanup.

A model can be removed if another model dominates it.

A model can also be removed if it is too close to a neighbor and does not provide enough unique value.

This avoids flooding users with barely different models that only look important on paper.

---

# 12. Reading a Failed Prediction

Consider this failed strict dominance example:

```json
{
  "reason": "StrictDominanceReplacement",
  "predicted": {
    "sizeGiB": "2.41",
    "kld": 0.021967,
    "lineKldAtPredictedSize": 0.022351,
    "gainOverLine": 0.000384
  },
  "actual": {
    "sizeGiB": "2.41",
    "kld": 0.022565,
    "lineKldAtActualSize": 0.022351,
    "gainOverLine": -0.000214,
    "sizeMissBytes": 0,
    "kldMiss": 0.000214
  },
  "accepted": false
}
```

The prediction said:

```text
This hybrid should be smaller than or equal to the anchor
and lower KLD than the anchor.
```

The actual benchmark said:

```text
Size was fine.
KLD was not.
```

The candidate missed by:

```text
0.000214 KLD
```

That is not catastrophic. It is not a sign that the predictor is useless.

But it failed the reason it was selected.

So MagicQuant rejected it.

That is the system working correctly.

Now consider the interior example:

```json
{
  "reason": "InteriorSubspaceDiscovery",
  "predicted": {
    "sizeGiB": "3.78",
    "kld": 0.001210,
    "lineKldAtPredictedSize": 0.001608,
    "gainOverLine": 0.000398
  },
  "actual": {
    "sizeGiB": "3.78",
    "kld": 0.001884,
    "lineKldAtActualSize": 0.001608,
    "gainOverLine": -0.000276
  },
  "accepted": false
}
```

The prediction said:

```text
This should beat the local line between Q6_K_XL and Q8_0.
```

The actual benchmark said:

```text
It did not beat the line.
```

So it was rejected.

Again: not because the prediction was “bad” in a simplistic numeric sense, but because the candidate failed the outcome contract.

That is the correct validation philosophy.

---

# 13. Practical Accuracy vs Perfect Accuracy

MagicQuant’s prediction system should be described as **practically accurate**, not perfect.

That wording is important.

Perfect accuracy would mean the engine correctly predicts every local ordering, every nonlinear inversion, every microscopic KLD difference, and every sub-4-bit oddity.

That is not the claim.

Practical accuracy means:

```text
The prediction engine is accurate enough to identify the candidates worth physically validating,
and wrong predictions are filtered by real benchmark outcome checks before publication.
```

That is a stronger engineering claim.

It is not mystical.

It is not pretending to know more than it knows.

It is a very fast, empirically grounded way to avoid wasting benchmark time on combinations that almost certainly will not matter.

---

# 14. The Q6_K vs Q8_0 Attn Output Example

The `attn_output` behavior in Qwen3 4B 2507 Instruct is a perfect example of why MagicQuant is honest about its limits.

In isolation, Q8_0 measurably beat Q6_K for `attn_output`.

So the additive backbone learned:

```text
attn_output: Q8_0 is safer than Q6_K
```

That is the correct isolated conclusion.

But when tested inside full combinations, nonlinear behavior appeared.

Out of roughly 350 combinations where `attn_output` could have used Q6_K versus Q8_0:

```text
~65 combinations had Q6_K meaningfully beat Q8_0
~32 combinations had Q8_0 meaningfully beat Q6_K
the rest were Q8_0 wins or likely Q8_0 wins, but inside noise / not meaningfully separable
```

This is fascinating.

It proves that nonlinear combination effects exist.

It also proves why the prediction engine should not be described as omniscient.

The additive backbone is blind to some context-specific inversions because isolated samples cannot encode every surrounding combination. If Q8_0 wins in isolation, the backbone generally ranks Q8_0 above Q6_K for that group.

That creates a blind spot.

But here is the critical point:

> In measured pipeline behavior so far, this blind spot rarely changes the final survivor set in a way that justifies exploding the combinatoric search.

That is the practical win.

The prediction engine does not need to correctly worship every tiny local inversion.

It needs to find meaningful final models.

If the inversion is large enough to matter, sampling and fallback validation may surface nearby candidates. If it is too small, too noisy, or too local to affect the final frontier, MagicQuant should not waste massive compute chasing it.

This is the exact meaning of practical accuracy.

---

# 15. Why the Prediction Engine Can Be Wrong and Still Be Excellent

A prediction engine can fail in two broad ways:

```text
false positive:
    predicts a candidate is worth building,
    but real validation rejects it

false negative:
    fails to prioritize a candidate that might have been good
```

MagicQuant is designed to make false positives cheap and false negatives tolerable.

## 15.1 False Positives Are Contained

If MagicQuant predicts a candidate should beat an anchor or line, the candidate is built and benchmarked.

If it fails, it is discarded.

The cost is a build/benchmark attempt.

The final output remains clean.

This is why the pipeline can afford to be aggressive.

## 15.2 False Negatives Are Controlled by Practicality

False negatives are more subtle.

A perfect exhaustive search might find some combinations the predictor did not prioritize.

But if those combinations only differ by noise-level KLD, or do not change final ranking, or create redundant files, they do not matter much.

MagicQuant is not trying to prove no better microscopic combination exists.

It is trying to publish a meaningful set of models.

That is a different goal.

And it is the right goal for real users.

---

# 16. Noise-Level Differences

A huge part of MagicQuant’s philosophy is knowing when not to care.

Two candidates may differ by:

```text
0.000001 KLD
0.000005 KLD
0.000010 KLD
```

Depending on benchmark stability, corpus size, hardware behavior, and measurement conditions, that may not be meaningful.

This is why MagicQuant should avoid language like:

```text
Candidate A is absolutely superior to Candidate B because it is 0.000002 KLD lower.
```

Instead, the better language is:

```text
These candidates are within noise or near-noise range, so MagicQuant does not treat the difference as a meaningful survivor distinction.
```

This is also why the Rank-Safe Projection’s plateaus are valuable.

When the system lacks trustworthy ordering resolution, it can collapse candidates into tied regions instead of pretending tiny differences are destiny.

---

# 17. Sub-4-Bit Behavior

Sub-4-bit quantization is where the monsters start peeking out from the basement.

At very low bit depths, several things become more likely:

1. isolated measurements may become less transferable,
    
2. group interactions may become stronger,
    
3. KLD behavior may become less linear,
    
4. PPL and KLD may disagree more often,
    
5. quantization format quirks may matter more,
    
6. benchmark noise may become more visible,
    
7. tiny tensor-distribution differences may create surprising local wins.
    

MagicQuant handles this by combining:

```text
isolated physical measurements
bit-stress interaction correction
rank-safe projection
real benchmark validation
fallback attempts
final dominance filtering
meaningful spacing
```

So the system is not pretending sub-4-bit is clean.

It is building guardrails around the chaos.

---

# 18. The Local Linear KLD Line

One of the best concepts in MagicQuant is the local linear line.

Suppose there are two validated anchors:

```text
smaller anchor:
    size = S_small
    KLD  = K_small
    damage is higher

larger anchor:
    size = S_large
    KLD  = K_large
    damage is lower
```

For any candidate size between them, MagicQuant computes the expected straight-line KLD:

```text
t = (S_candidate - S_small) / (S_large - S_small)

line_KLD =
    K_small + t × (K_large - K_small)
```

The candidate must beat that line:

```text
actual_KLD(candidate) < line_KLD
```

This is brilliant because it avoids rewarding a candidate merely for existing between two known models.

A candidate between Q6_K and Q8_0 should not survive just because it is between them.

It should survive because it is better than the boring interpolation.

That is how MagicQuant finds nonlinear trade spaces worth showing.

---

# 19. What Makes a Hybrid “Earn Its Place”

A MagicQuant hybrid earns its place when it does at least one of the following:

## 19.1 Strict Dominance

It is no larger than an existing anchor and has lower real KLD.

```text
same or smaller size
lower damage
```

That is a clean win.

## 19.2 Near-Baseline Replacement

It is slightly larger than a smaller anchor but improves KLD enough to justify the size premium.

```text
small size increase
meaningful KLD improvement
```

## 19.3 Interior Discovery

It lands between two anchors and beats the local linear tradeoff line.

```text
better than expected for its size
```

## 19.4 Meaningful Gap Filling

It occupies a useful region of the size/fidelity frontier that would otherwise have a large jump.

```text
not redundant
not fake precision
actually useful to users
```

## 19.5 Final Survival

It survives dominance filtering and spacing cleanup.

```text
not eliminated by a better neighbor
not collapsed as meaningless clutter
```

That is what “earned its place” means.

Not “MagicQuant made a hybrid.”

Not “the prediction said it looked cool.”

Earned.

---

# 20. Why Baselines Can Beat Hybrids

MagicQuant is not biased toward hybrids.

Sometimes the best answer is a normal baseline quant.

That is fine.

In fact, that is part of the trust model.

If a pure Q6_K, Q5_K, Q4_K_M, Unsloth, llama.cpp, or other baseline artifact wins under equal evaluation conditions, MagicQuant should say so.

The goal is not to prove MagicQuant hybrids are always superior.

The goal is to provide a final downloadable set that is honest.

Sometimes the baseline already occupies the best tradeoff.

Sometimes a hybrid earns a new slot.

Sometimes a hybrid almost earns a slot but fails real validation.

All three outcomes are useful.

---

# 21. How the Prediction Engine Scales

The prediction engine scales because the expensive part is not proportional to the full combinatoric space.

If there are:

```text
G tensor groups
Q possible quant choices per group
```

A naive full search looks like:

```text
Q^G
```

That gets disgusting fast.

MagicQuant instead builds an isolation library closer to:

```text
G × Q
```

Then predicts large portions of the hybrid space using:

```text
additive damage
pairwise bit-stress correction
rank-safe projection
```

The result is not free.

But it is wildly more tractable.

This is why the new system can sample and build many times faster than the old approach. The prediction engine converts the search from “try everything” into “measure the local physics, infer the frontier, validate the likely winners.”

That is the whole game.

---

# 22. Repeatable Algorithm

This section describes the full repeatable process.

## Step 1: Scope the Prediction Bucket

Predictions are scoped by:

```text
model
architecture family
imatrix identity
benchmark category
active tensor groups
```

Do not mix unrelated contexts unless intentionally building a cross-model research layer.

## Step 2: Build Pure Baselines

Build and benchmark pure baselines such as:

```text
Q8_0
Q6_K
Q5_K
Q4_K_M
IQ4_XS
external/custom baselines where applicable
```

These become anchor points for final comparison.

## Step 3: Build Exact-Blanket Anchors

Build exact-blanket carrier states where active tensor groups are held at native/exact precision under a carrier.

The key anchor is the Q8 exact blanket.

This provides the size and KLD reference environment for isolated group probes.

## Step 4: Build Single-Group Isolations

For each active tensor group and each relevant quantization target:

```text
start from Q8 carrier exact blanket
force exactly one group to target quant
leave all other groups exact
benchmark KLD/PPL/size
store result
```

This creates:

```text
D(g, q)
```

The isolation damage library.

## Step 5: Generate Candidate Configurations

Generate candidate hybrid configurations from the allowed quant choices.

Discard configurations that are impossible, unsupported, missing required mappings, or outside search policy.

## Step 6: Compute Effective Baselines

For each candidate and group:

```text
effective = override if present else base quant
```

Normalize external baselines to their built-in standard baseline when needed for isolation lookup.

## Step 7: Predict Additive Damage

Compute:

```text
A(c) = Σ D(g, e_g(c))
```

If required isolation data is missing, mark the candidate as not safely predictable.

## Step 8: Predict Size

Compute predicted size from base-only size anchors and isolated size deltas.

If required size anchors are missing, mark the candidate as unsafe for size-based selection.

## Step 9: Fit Interaction Correction

Using existing real benchmark rows in the current bucket:

1. Try candidate bit-stress thresholds.
    
2. Compute cross-terms.
    
3. Fit `α` and `β`.
    
4. Choose the threshold with the best error.
    
5. Fall back to additive-only behavior if too few rows exist.
    

## Step 10: Compute Interaction Estimate

Compute:

```text
Y(c) = α × A(c) + β × X_B(c)
```

## Step 11: Apply Rank-Safe Projection

Sort predictable rows by additive damage, interaction damage, and size.

Apply PAVA.

The resulting projected value becomes:

```text
PredictedKld(c)
```

## Step 12: Select Candidate Attempts

Use prediction to identify candidates for:

```text
strict dominance replacement
near-baseline replacement
interior subspace discovery
```

Keep only a limited number of fallback attempts per anchor/window.

## Step 13: Build and Benchmark Candidates

For each selected candidate:

```text
quantize
benchmark
load real snapshot
evaluate actual size/KLD against expected outcome
```

## Step 14: Accept or Reject by Outcome

Do not accept based on prediction error.

Accept only if the real benchmark satisfies the reason it was selected.

## Step 15: Final Dominance and Spacing

Merge accepted candidates with existing anchors.

Remove dominated models.

Collapse meaningless near-neighbor clutter.

Export the final survivor set.

---

# 23. Pseudocode

```text
for each scoped model/imatrix bucket:

    active_groups = detect_active_tensor_groups(model)

    pure_baselines = benchmark_pure_baselines()

    q8_exact_blanket = benchmark_exact_blanket(base=Q8_0)

    isolation_library = {}

    for group in active_groups:
        for quant in candidate_quants:
            sample = benchmark(
                base=Q8_0,
                all_groups=exact,
                override[group]=quant
            )

            isolation_library[group, quant] = sample.KLD

    candidates = generate_hybrid_candidates()

    for candidate in candidates:

        additive = 0

        for group in active_groups:
            effective = candidate.override[group] or candidate.base

            if effective is Q8_0 or exact:
                damage = 0
            else:
                damage = isolation_library[group, effective]

            additive += damage

        candidate.additive_kld = additive
        candidate.predicted_size = predict_size(candidate)

    fit = fit_bit_stress_interaction(candidates_with_real_truth)

    for candidate in candidates:
        cross = compute_pairwise_bit_stress_cross_term(candidate)
        candidate.interaction_kld = fit.alpha * candidate.additive_kld + fit.beta * cross

    candidates.predicted_kld = rank_safe_projection(
        order_by = additive_kld,
        values   = interaction_kld
    )

    attempts = select_predicted_candidates(candidates)

    for attempt in attempts:
        real = build_and_benchmark(attempt)

        if satisfies_expected_outcome(real, attempt.reason):
            accept(real)
        else:
            reject(real)

    survivors = final_dominance_and_spacing_pass(accepted + pure_baselines)

    export(survivors)
```

---

# 24. How to Explain the Accuracy

The most honest way to describe MagicQuant prediction accuracy is not:

```text
It predicts KLD perfectly.
```

It is:

```text
It predicts the useful search frontier with high practical accuracy, then validates every selected survivor with real benchmarks.
```

The accuracy should be reported in several layers.

## 24.1 Numeric Error

Useful metrics:

```text
MAE
RMSE
max absolute error
mean signed error
relative error
```

These answer:

```text
How close were predicted KLD values to actual KLD values?
```

## 24.2 Rank Accuracy

Useful metrics:

```text
pairwise order accuracy
Spearman correlation
Kendall tau
rank distance
within-one-rank count
within-two-ranks count
within-five-ranks count
```

These answer:

```text
Did the predictor order candidates correctly?
```

For MagicQuant, rank accuracy is often more important than raw KLD error because candidate selection depends heavily on frontier ordering.

## 24.3 Outcome Accuracy

This is the most important pipeline metric.

Useful metrics:

```text
accepted predicted candidates
rejected predicted candidates
strict dominance success rate
near-baseline success rate
interior discovery success rate
fallback success rate
final survivor count
dominance eliminations
spacing eliminations
```

These answer:

```text
Did predictions lead to real useful models?
```

That is the metric users should care about most.

## 24.4 Noise-Aware Accuracy

MagicQuant should also distinguish:

```text
wrong but inside noise
wrong and measurable
wrong and final-rank-changing
```

These are not the same.

A prediction that reverses two candidates separated by microscopic KLD is not as important as one that incorrectly replaces a major anchor.

This is why the docs should avoid treating all prediction misses as equal.

---

# 25. Recommended Language for Documentation

Use language like:

```text
MagicQuant is isolation-grounded and rank-safe.
```

```text
The predictor is not omniscient; it is a practical search-space reduction engine.
```

```text
Predictions nominate candidates. Real benchmarks decide survivors.
```

```text
The additive isolation backbone provides rank discipline.
```

```text
The bit-stress interaction estimator improves numeric fit in low-bit regions.
```

```text
The rank-safe projection prevents the interaction estimator from inventing unstable local ordering.
```

```text
Plateaus are uncertainty, not failure.
```

```text
MagicQuant does not fight for pennies.
```

```text
A hybrid survives only if it earns a meaningful size/fidelity slot.
```

Avoid language like:

```text
MagicQuant always finds the best possible quant.
```

```text
The predictor is always correct.
```

```text
Every lower KLD value is meaningful.
```

```text
Hybrids are always better than baselines.
```

Those claims are weaker because they are too absolute.

The real story is better anyway.

MagicQuant is impressive because it is honest.

---

# 26. Beginner Explanation

Imagine you are trying to find the best way to compress a model.

The impossible way is:

```text
Try every possible compression choice for every part of the model.
```

That explodes instantly.

MagicQuant does something smarter.

It asks:

```text
What happens if I compress only this one part?
What happens if I compress only that one part?
What happens if I do this for every major part?
```

Then it builds a map of how each part behaves.

Once it has that map, it can estimate what will happen when multiple parts are compressed together.

Most of the time, damage adds up in a predictable way.

Sometimes low-bit compression makes groups interact weirdly, so MagicQuant adds an interaction correction.

Then it forces the final prediction to respect the stable order learned from isolated measurements.

Finally, it tests the best-looking candidates for real.

If the real test fails, the candidate is thrown away.

That is MagicQuant in one breath.

---

# 27. Advanced Explanation

MagicQuant treats hybrid quantization as a structured prediction problem over tensor-group effective baseline assignments.

The core assumption is that the measured KLD contribution of a group under a Q8-carrier native-exact blanket provides a transferable marginal damage estimate for that group’s effective quantization state.

The first estimator constructs an additive functional over group states:

```text
A(c) = Σ_g D(g, e_g(c))
```

This additive functional is used primarily as a rank prior.

A second estimator augments the additive functional with a low-bit pairwise stress term:

```text
Y(c) = αA(c) + βΣ_{g<h}D_gD_h max(0, B-b_g)max(0, B-b_h)
```

The threshold `B` and coefficients are selected per bucket against available benchmark truth.

The final estimator performs an isotonic projection of `Y` onto the partial order induced by `A`, producing the closest monotone sequence to the interaction estimator while preserving additive rank discipline:

```text
Z* = argmin_Z Σ_i (Z_i - Y_i)^2
subject to Z_i ≤ Z_j whenever A_i ≤ A_j in sorted order
```

PAVA computes this projection efficiently.

The projection converts local estimator disagreement into plateaus. These plateaus are semantically useful because they express uncertainty rather than overfitted micro-rankings.

Candidate selection then uses the projected KLD and predicted size to nominate strict replacements, near-baseline replacements, and interior better-than-linear candidates.

Every candidate is physically built and benchmarked before acceptance.

The predictor therefore controls benchmark allocation, not final truth.

---

# 28. Final Summary

MagicQuant’s prediction engine is powerful because it combines three ideas that fit the real physics of quantized models:

1. **Monotonic degradation is overwhelmingly common.**  
    Higher-fidelity quantization usually produces equal or lower KLD when the rest of the model is held constant.
    
2. **Isolated tensor-group measurements are highly informative.**  
    Single-group Q8-carrier exact-blanket probes provide a reusable physical map of marginal group damage.
    
3. **Rank-safe projection prevents fake certainty.**  
    The system uses interaction correction where helpful, but forces final predictions to respect the stable additive isolation order unless uncertainty collapses values into plateaus.
    

The result is not a perfect oracle.

It is something better for real engineering:

```text
a fast, honest, empirically grounded search engine
that spends benchmarks where they matter
and lets real benchmark validation decide what survives.
```

MagicQuant does not need to be omniscient.

It needs to be right where it counts.

And when it is wrong, it needs to fail safely.

That is exactly what the pipeline is built to do.