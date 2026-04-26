# MagicQuant (v1.0)

The following documentation is for version 1 of MagicQuant. Please note that version 1 was truly a prototype, heavily flawed, and often misunderstood.

The goal of version 1 was never to build a superior quantization tactic, but to do the following:
1.) Judge quantization tactics like llama.cpp or Unsloth (even if Unsloth wasn't used, it was built for such ideals). 
2.) Find and validate that weird combinations exist that create non linear recovery in damage.

There was side alternative goals too, but it's important to preface that I am leaving version 1 documentation up for transparency, knowledge share, and understanding. As failure often is more powerful to learn from than success. Especially when the failure looked deceptively successful!

This `README.md` documentation will go over what MagicQuant (v1.0) was trying to accomplish, what it was doing right or wrong, and more. The original documentation will be provided in this wiki as well, but it's important to note I'm not altering the original documents that're archived, thus you must take into account what this document says here to understand the flaws in what was originally written.

Don't get me wrong, MagicQuant v1.0 was really cool, interesting, and had fascinating success, but what will be shown here is also the flaws, what I learned through mistakes, and more.

---

# History of MagicQuant v1.0

Before I gave this project the name "MagicQuant" it all started because everyone in my bubble kept saying "MXFP4_MOE on GGUF is bad". This led me to say, "where's your benchmarks?" but nobody did that!

So, I did it and discovered, "oh this is bad"! But I kept playing around with it for fun just to prod it a bit more. Not out of a love for MXFP4, but out of curiosity. Because in theory it should be different, not "worse". Though GGUF MXFP4 is not necessarily real MXFP4 mind you.

But by pure accident I played with some grouping and found an MXFP4 mix that was a true lottery ticket of a quant. At least that's what was perceived as things got weird in a good way very fast!

I would then post about this finding and the weird hybrids I created on [Reddit (click here)](https://www.reddit.com/r/LocalLLaMA/comments/1ozh8py/mxfp4_hybrid_dense_models_ready_to_share_near/).

But to me it was fascinating, but it made me wonder, "are there more secret sauce hybrids that exist?" Additionally for quite some time, I've been annoyed with people post Q8, Q6_K, and so on without benchmarks easily visible when llama.cpp makes it pretty easy, fast, and accessible. So I wanted to make a benchmarking judge like system that also validated if specific tensor groups could achieve combinations that really made a difference. This is how MagicQuant v1.0 was born as I would then begin to test my idea in a framework I created, as manual instinct based iterations were no longer sufficient. Thus I made MagicQuant and again [posted on Reddit (click here)](https://www.reddit.com/r/LocalLLaMA/comments/1piasv8/magicquant_hybrid_evolution_gguf_tps_boosts/).



# What was v1.0 MagicQuant?

MagicQuant v1.0 was an evolutionary search system that would build hybrid mixed models with different tensors in different groups depending on their benchmarks. This required a lot of sampling, an adaptive predictive engine, and more.

This found models that had better TPS, lower PPL, and often were smaller or it'd find weird sub spaces between pure baselines.

What was found in version 1 isn't necessarily purely "wrong" but it wasn't "right" without the new context of what was learned and why much of it isn't what it seemed.

# v1.0 Flaws

First and most importantly, v1.0 (and obviously myself) didn't understand tensor level quantization the way I do now. I would literally build quants (even if automated) effectively like so:
```bash
llama-quantize \
  --tensor-type token_embd.weight=Q6_K \
  --tensor-type output.weight=MXFP4 \
  --tensor-type 'router.*'=Q6_K \
  --tensor-type 'gate.*'=Q6_K \
  "Path_To_F16_GGUF.gguf" \
  "Path_To_GGUF.gguf" \
  mxfp4_moe
```

If you understand quantization well enough like I do now, you can see the horror of what I was doing in all previous experiments. I effectively did this even when targeting large groups with multiple tensors. Why is this such an issue? Well let me explain the horror:

For example, MagicQuant has multiple tensor groups it assigns tensors too. **But every tensor in that group got quantized to the same quant!** Hint hint, this isn't okay. In fact, PPL didn't detect this in my testing, only when KLD was added in version 2 was I signaled that something was incredibly off with my models.

There's a ton of things about tensors in general. Some are 1 dimensional and thus shouldn't be touched, some are more sensitive than others. My goal was never to be tensor level like that, I didn't realize this is what I was doing with my quantization tactic. And mind you, KLD is the primary favored metric in v2.0 but PPL isn't necessarily "worse" or "nothing" it measures something different, but it as a bad primary metric for what MagicQuant was trying to achieve. And note, v2.0 worked hard to resolve everything being discussed.

But PPL chasing was also making models do what I call, "pretending to be smart". Because lower PPL means they can spit out what they want to spit out closer to the original model, but it's reasoning was deeply impacted most likely. As KLD was enormous after reviewing the original MagicQuant models with better KLD metrics.

### How Quantization Should Look

If you look at how models get quantized by llama.cpp for example. Some tensors they never let go below Q6_K, some are never touched, some are quantized. Llama.cpp does good work, but it's not as specialized as something like Unsloth. Unsloth models often have a wider range of quants to protect the brain while quantizing.

A great example of a well quantized model would have a mix of quants to tensors in groups and we can look at Unsloth Dynamic 2.0 GGUF models from their Qwen3 4B series. THe following is the Unsloth Qwen3 4B 2507 Instruct model quantized to Q3_K_XL:

| Tensor Group | Unique Final Quant Types |
|---|---|
| `embeddings` | `Q6_K` |
| `attn_q` | `IQ3_XXS`, `IQ4_XS`, `Q3_K`, `F32` |
| `attn_kv` | `IQ3_XXS`, `IQ4_XS`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `F32` |
| `attn_output` | `Q4_K`, `F32` |
| `ffn_up_gate` | `IQ3_S`, `IQ4_XS`, `Q3_K`, `F32` |
| `ffn_down` | `Q4_K`, `Q5_K`, `Q6_K`, `F32` |

Note missing tensor groups means the model didn't have it. This model also did cool stuff like blending the LM_Head into other places.

Anyways, you can see there's an insanely diverse amount of quantizations across the board by Unsloth. Their system is obviously detecting sensitivity and more.

But the fatal flaw MagicQuant v1.0 made was it'd do the following:

| Tensor Group  | Unique Final Quant Types |
| ------------- | ------------------------ |
| `embeddings`  | `Q6_K`                   |
| `attn_q`      | `IQ4_XS`                 |
| `attn_kv`     | `Q5_K`                   |
| `attn_output` | `Q4_K`                   |
| `ffn_up_gate` | `Q3_K`                   |
| `ffn_down`    | `Q4_K`                   |

This caused massive hidden damage that became detectable with KLD. But it also was primary insight into what would become v2.0 to truly pick up the title of "be the judge, not the quant". Because my goal was never to try an beat Unsloth Dynamic. This was a very misunderstood thing about the original MagicQuant system. I wanted to decide what quants were best, use what was best, not be the quantization itself. Which is why philosophically MagicQuant v2.0 would flip everything upside down.
## Tokens Per Second Is A Difficult Metric

Another thing v1.0 did was take into account TPS. The issue is that TPS boons are not universal across hardware the way I initially assumed. Thank you to those who privately shared metrics with me to compare and talk about it! Many validated the TPS boons, but there were some who said, "this makes it worse for me not better". This immediately caught my attention. Was it PCIE bus differences on larger models or something else?

But nope, a pretty simple test resolved this. I had a 3090 and the other user had a 5090. We both tested a MagicQuant model that fit in a single GPU in full. In our simple tests, he got worse TPS while I got better. Different hardware architecture makes TPS gains from quantization not universal. That is obvious in hindsight but I didn't think the difference would be so large.

This is why in v2.0 mind you TPS is a dropped metric. I've paved the road for the future to potentially re-explore this but for now, it's off the table as it's not the goal either.

Faster for me may not be faster for you. This was another flaw.


# The Search Space Flaw

There was many other flaws I could go on for days for, but the final and most fatal flaw of MagicQuant v1.0 was the lack of knowledge I have now on the data itself.

Though v1.0 didn't fully figure out the fullness of what I'm about to say, it sang enough to me that I realized, v1.0 was not just flawed but the tactics being utilized are likely overkill, wrong, and more importantly "not worth it".

Once a proper kld and quantization system was built in v2.0 where I built significantly more confidence in my benchmarking, could test/sample 3x-9x faster, I simply downloaded tons of models and architectures and built massive amounts of sample data of significant amount of combinations of hybrids.

Though MagicQuant was flawed, both v1.0 and v2.0 did confirm the existence of non linear weird combinations that caused decreased damage in both KLD and PPL that was not predictable (in any way I could formalize) without building real hybrid combo's to find this rare mix. Which MagicQuant v1.0 though flawed excelled at.

Real numbers and data provided elsewhere, but here I'll just tell you the vibe. Overwhelmingly the world is more linear than not. Like overwhelmingly so. And recorded in my data is insanely low amounts of sub space where mixes beat real world expectations (aka a group tensor does better with Q6_K than Q8_0 for example when both were not measured that way in isolation).

And these examples do exist. Take Qwen3 4B Instruct 2507 again and if you create significant amounts of hybrid combo's. There's a real number of combinations where that same combination when it had Q6_K versus Q8_0 that Q6_K had a better KLD than Q8_0 and was "better" in a meaningful measurement, AKA it was outside the range of noise. Now the majority of the situations Q8_0 still was superior, but there was real recorded instances where Q6_K did better.

But here's what's crucial to understand. "Meaningful" I'm defining as outside of measured noise, not "meaningful to humans". The difference was so small that if we could find these mixes early, that'd be cool and all, but they're still not worth the combinatoric space, search, or time it takes to even consider them. Seldom if ever is it chosen as a final candidate anyways. 


# The Takeaway

The quantization space is much more linear. Q8_0 is likely to do better than Q6_K (if Q8_0 was measured to be better in isolation in X group vs Q6_K I mean). This seems obvious, but MagicQuant v1.0 spent a long time trying to brute force reality to see if this truth wasn't as strict as was made out to be.

And MagicQuant v2.0 assisted in confirming that this weirdness exists, but it also put the nail in the coffin for the idea of chasing these weird combo's through an evolutionary system.

Even if MagicQuant v1.0 didn't have the design flaws, it still showcased interesting things. PPL is weird, it's not as linear as KLD mind you. PPL is very very weird.

Additionally shown in both versions is that measured quants that's worse in isolation can in combination with other tensor combinations perform better than expected and to the point that it can surpass a close neighbor in some scenarios.

I hope this helps you understand the flaws of v1.0 and why it was completely abandoned. I never wanted to be a quantizer, I want to be the judge and v2.0 hit that desired goal. But as you look through v1.0 documentation (if you do) keep this in mind that the documentation was also written without the knowledge written here.

[Original `README.md` and attached documentation to MagicQuant v1.0](./old/README.md)