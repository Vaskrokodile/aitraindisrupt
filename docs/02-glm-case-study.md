# Report: How Far RL Has Pushed the GLM Family (and What It Means for Small-Model RL)

*Scope note / timeline. As of the sources available (through ~Aug 2026), the GLM lineage relevant here is: GLM-4-0414 / GLM-Z1 (Apr 2025) → GLM-4.5 / 4.5-Air (Jul 2025) → GLM-4.6/4.7 → GLM-5 (early 2026, arXiv 2602.15763) → GLM-5.1 → GLM-5.2 (Jul 2026) → **GLM-5.3 (Aug 14, 2026)** — the most RL-informative release. Numbers below are from search-result excerpts of primary sources (Z.ai blogs, arXiv abstract/PDF, slime README, SAO paper coverage) plus credible secondary write-ups (MarkTechPost, Baseten, W&B). Flagged where uncertain.*

---

## 1. The GLM RL pipeline: how much is RL vs SFT/distillation?

### GLM-4.5 (arXiv 2508.06471) — "Expert Model Iteration"
Post-training is a **two-stage SFT+RL sandwich**:

- **Stage 1 (Expert Training):** small cold-start SFT with long-CoT data → specialized RL to build three experts (Reasoning, Agent, General chat).
- **Stage 2 (Unified Training):** SFT **self-distillation** merges experts into one hybrid-reasoning model. So RL builds capability; distillation consolidates it — RL is the capability engine, SFT is packaging.
- Reasoning RL specifics (from the Z.ai GLM-4.5 blog): "a **single-stage RL over the full 64K context with a difficulty-based curriculum**, which we found superior to progressive scheduling," plus "dynamic sampling temperatures to balance exploration–exploitation and adaptive clipping for robust policy updates." Notably: "Although the RL curriculum targets a limited set of verified tasks, the resulting gains **transfer to adjacent abilities such as general tool use**."
- Result: 355B-A32B MoE → 91.0 AIME24, 79.1 GPQA, 72.9 LiveCodeBench, 64.2 SWE-bench Verified, 70.1 TAU-bench. The Air variant (106B-A12B) retains most of it (89.4 AIME24) — i.e., RL gains distill down ~3.3× parameter reduction with modest loss.

### GLM-5 (arXiv 2602.15763, 744B-A40B)
- Post-training: "**sequential RL pipeline — Reasoning RL → Agentic RL → General RL**" with "**On-Policy Cross-Stage Distillation**" to prevent catastrophic forgetting. SFT is reduced to a substrate; the report says they "moved beyond standard SFT."
- Infrastructure: **slime**, an async RL framework decoupling generation (SGLang) from training (Megatron). slime's README states it is "the RL framework behind GLM-5.3, GLM-5.2, GLM-5.1, GLM-5, GLM-4.7, GLM-4.6, and GLM-4.5" — i.e., the entire modern GLM line is RL-productionized.
- GLM-5 report explicitly proposes "**asynchronous agent RL algorithms**" for long-horizon interactions.

### GLM-5.2 → SAO (the key algorithmic event)
- GLM-5.2's agentic RL used **SAO (Single-rollout Asynchronous Optimization)**, a Zhipu+Tsinghua KEG paper released Jul 15, 2026. Findings reported:
  - Vanilla GRPO **collapses at ~160 async training steps**; SAO trains stably ~1,000 steps.
  - Mechanism: abandons group sampling (one rollout/prompt), adds **DIS** (double-sided token-level clipping that *masks* out-of-trust-region tokens rather than bounding ratio), **re-introduces a critic** (frozen-attention value net — "put the critic back into RL"), and **Skip-Observation GAE** (excludes environment-observation tokens from advantage computation).
  - Numbers (Qwen3-30B-A3B experiments): AIME 2025 — base 72.1 → GRPO 84.2 → GRPO+DIS 86.5 → **SAO 97.3**. SWE-bench Verified — base 23.0 → GRPO+DIS 27.0 → **SAO 29.8**. (Secondary-source numbers; verify against the SAO paper itself.)

### GLM-5.3 — the cleanest "RL-only" industrial datapoint
- **Same base model as GLM-5.2; every gain is post-training.** ~1 extra month of scaled RL: more synthesized executable environments, longer-horizon tasks (some = days of engineer work), stronger verifiers.
- Reported deltas (MarkTechPost / Z.ai docs): Z.ai Code Bench **+50%** (to 31.4% at ~50k output tokens vs Claude Opus 4.8's 29.5% at 120k); **Terminal-Bench 3.0: 4.6 → 28.3**; DeepSWE v1.1: 46.2 → 66.9; Agents' Last Exam (CLI): 23.8 → 28.5; CyberGym: 77.2 → 84.5; ExploitBench: 24.4 → 54.4; ExploitGym tasks solved in 2h: 29 → 105.
- Emergent behavior: cyber capability "developed faster than expected" — exploitation-chain reasoning compounded with scale. Gains grow with horizon length (consistent with RL teaching process, not facts).
- Anti-reward-hacking detail (Baseten write-up): verifiers must pass three tests — oracle-solution gets reward, do-nothing gets none, incomplete gets none; solver trajectories used to close reward shortcuts. Environments are **synthesized end-to-end by pipelines** (incl. synthesized rewards for a subset).

### GLM-Z1 (Apr 2025, 32B and 9B dense)
- GLM-Z1-32B-0414: "cold start + **extended reinforcement learning**" on math/code/logic, plus pairwise-ranking general RL. **GLM-Z1-9B** was trained "with the aforementioned series of techniques" — i.e., Zhipu applied the same RL recipe to a 9B and claimed size-class SOTA, though no RL-vs-distillation ablation for the 9B was published. This is direct evidence a lab ran extended RL (not just distillation) at 9B successfully — contrary to DeepSeek's conclusion (below).

---

## 2. Quantified RL gains across the industry

| Model/Result | RL delta | Source |
|---|---|---|
| DeepSeek-R1-Zero (pure RL, zero SFT, 671B base) | AIME24 pass@1 **15.6% → 71.0%** (86.7% w/ maj-vote) over "thousands of RL steps" | arXiv 2501.12948 |
| SAO vs GRPO (30B-A3B, async agentic) | AIME25 84.2 → 97.3; SWE-V 27.0 → 29.8 | SAO paper coverage |
| GLM-5.3 vs 5.2 (same base, RL-only) | Terminal-Bench3 4.6→28.3; Code Bench +50%; ExploitBench 24.4→54.4 | Z.ai/MarkTechPost |
| DeepSeek small-model finding | On Qwen2.5-32B, **distilling R1 beats doing RL on it** — R1-Distill-Qwen-7B hits 55.5 AIME24, beating QwQ-32B-Preview | DeepSeek-R1 paper §2.4 |
| Open-RS (1.5B, constrained compute) | R1-Distill-Qwen-1.5B + GRPO, 4×A40, 24h, ~$42 → AIME24 46.7% (>o1-preview 44.6) | arXiv 2503.16219 |

**RL-vs-capability literature (important caveat):** Yue et al.–style analyses (e.g., arXiv 2505.14216, "RL vs. Distillation") find RLVR raises **pass@1 but often not pass@k** — it concentrates probability on already-solvable problems and can *hurt* the hardest ones; it improves "accuracy, not capability," whereas distillation can add capability *only when introducing new knowledge*. Counter-literature (Setlur et al., Sun et al., Liu et al.) shows RLVR can solve previously-unsolvable problems when train/test difficulty is matched — so the "ceiling" claim is contested, and the GLM-5.3 emergent-cyber result is industrial evidence *against* a hard ceiling when environments keep scaling.

---

## 3. What Zhipu says about limits

- **No admission of RL saturation** — the opposite. GLM-5.3 blog: "Scaling post-training is all we did." GLM-5 repo: "RL aims to bridge the gap between competence and excellence... enabling more fine-grained post-training iterations." Their bottleneck moved to **environments**: "much of the difficulty in scaling post-training moves from the model to the environment" — hence automated environment synthesis.
- **Forgetting is their acknowledged RL limit**: GLM-5 uses on-policy cross-stage distillation specifically because sequential RL erodes earlier skills. RL is leaky — you must distill between stages.
- **Stability was the real wall**: GRPO collapsing at ~160 async steps is the published failure mode that forced SAO. The constraint wasn't reward quality or data — it was algorithmic stability under asynchrony.
- **Small models**: GLM-Z1-9B implies RL works at small scale in their hands, but Zhipu has published no small-model RL-vs-distillation ablation. DeepSeek's opposite published finding (distill > RL at 32B) remains the field's standard caveat.

## 4. The empirical "RL ceiling" — what broke through

Documented breakthroughs when RL plateaued/collapsed:
1. **Algorithmic**: async GRPO collapse → fixed by single-rollout + critic + strict clipping (SAO). Compute alone wouldn't fix this.
2. **Environment/reward scaling**: GLM-5.2→5.3 — same base, same stack, more realistic verifiable environments → largest gains on longest-horizon tasks. Bottleneck was task supply, not the model.
3. **Base-model capability**: R1-Zero needed a 671B base with latent reasoning to reach 71% AIME; DeepSeek found small bases can't RL their way to distilled-level reasoning — the base is the floor RL polishes.
4. **Cold-start data**: R1 needed cold-start SFT to fix R1-Zero's readability/language-mixing — pure RL converged to functional but degenerate outputs.
5. **Forgetting mitigation**: on-policy distillation between RL stages (GLM-5) — sequential RL without it plateaus via skill loss.

---

## LESSONS FOR SMALL-MODEL RL (1–7B, constrained compute)

1. **RL on a small model raises pass@1, not the ceiling.** The strongest replicated finding (Yue et al.; DeepSeek's own ablation) is that RLVR concentrates probability mass on problems already in the base distribution. If your goal is *new* capability at 1–7B, pure RL is the weaker lever — distillation injects knowledge RL can't discover. **But**: the counter-evidence says matched-difficulty curricula can push pass@k too — so design the difficulty band carefully (this is exactly GLM-4.5's "difficulty-based curriculum beats progressive scheduling" insight, at 64K context).
2. **Your base model is the ceiling.** R1-Zero worked because V3-Base already contained latent reasoning. Before investing RL compute, measure pass@k on the base — that approximates what RL can harvest. A better/cheaper intervention may be mid-training (GLM-5's approach) rather than post-training.
3. **The Open-RS recipe is the constrained-compute template**: take the strongest *distilled* small model (not the raw base), then short GRPO runs — 1.5B → 46.7% AIME24 for ~$42/24h on 4 GPUs. Distill first, polish with RL second. Don't RL a raw base at small scale.
4. **Environments > algorithm at the frontier.** Zhipu's 5.3 thesis — the bottleneck shifted to executable, verifiable, realistic tasks with hack-proof rewards (oracle/nothing/incomplete verifier tests). For small models, investing in *verifiable task generation* likely yields more than another PPO variant.
5. **Stability tricks transfer down**: token-level masking of off-trust-region samples (SAO's DIS), skipping observation tokens in advantage (Skip-Observation GAE), dynamic temperature for exploration, and adaptive clipping are all cheap to implement at 7B and address the dominant failure mode — collapse before enough steps.
6. **Watch for forgetting**: if you run staged RL (math → code → agentic), interleave on-policy distillation or rehearsal; GLM-5 treats this as mandatory, not optional.
7. **Emergent behaviors appear with scale of training, not just model size** — GLM-5.3's exploitation-chain planning was unplanned. At small scale, long-horizon multi-step tasks (where trajectories, not single answers, are rewarded) are where RL plausibly teaches something SFT data can't easily encode — that's the most promising niche for "novel RL-like" algorithms.
8. **Honest uncertainty**: no published controlled study isolates "RL ceiling vs model size" cleanly; Zhipu's 9B RL success vs DeepSeek's 32B distill>RL finding are both single-lab claims. Treat "RL can't expand small-model capability" as a prior, not a law — the strongest falsification path is exactly what you'd be researching.

**Key sources**: arXiv 2508.06471 (GLM-4.5); arXiv 2602.15763 (GLM-5); z.ai/blog/glm-5.2, /glm-5.3, /glm-4.5; github.com/THUDM/slime; SAO paper (Jul 2026, Zhipu+Tsinghua KEG); arXiv 2501.12948 (DeepSeek-R1); arXiv 2505.14216 (RL vs Distillation); arXiv 2503.16219 (Open-RS); MarkTechPost 2026-08-14; baseten.co/blog/glm-53; wandb.ai GLM-5.2 report.

*Couldn't fetch full arXiv HTML for GLM-5 §RL details and the SAO paper's own tables (secondary numbers flagged).*
