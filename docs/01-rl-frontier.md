# Frontier Report: RL for LLMs (2024–2026), with Small-Model Emphasis

*Scope note: arxiv IDs below were verified via search where noted; others are cited
with caution flags. This is a map for novelty-claim grounding, not an exhaustive survey.*

---

## 1. Algorithm lineage: PPO → GRPO → the 2025 variant soup

**Verified / solid:**

- **GRPO** (Shao et al., DeepSeekMath, arXiv 2402.03300): removes the critic, uses group-mean-normalized advantage, token-level importance ratios, KL-to-reference. Basis of DeepSeek-R1 (2501.12948).
- **DAPO** (arXiv 2503.14476, ByteDance/Tsinghua; NeurIPS 2025): four changes — Clip-Higher (decoupled asymmetric clip bounds ε_low/ε_high to fight entropy collapse), Dynamic Sampling (drop all-correct/all-wrong groups that give zero advantage), Token-level policy-gradient loss, Overlong Reward Shaping. Achieved 50 AIME'24 on Qwen2.5-32B vs R1-Zero-Qwen-32B, in ~50% steps. **Most of its gains are training-stability/data-quality, not new theory** — but it became the de-facto open recipe.
- **Dr. GRPO** (in "Understanding R1-Zero-Like Training," arXiv 2503.20783, Sea AI Lab; ICML 2025): removes GRPO's length-normalization and std-normalization biases which artificially inflate response length, especially on *wrong* answers. Same paper showed **Qwen2.5 bases already reason without templates** and DeepSeek-V3-Base already shows "aha" — i.e., much of R1-Zero's apparent magic is pretraining elicitation. Minimalist recipe: Qwen2.5-Math-7B, MATH L3-5, 27 hrs on 8×A100 → 43.3% AIME. Key result for small-compute work.
- **GSPO** (arXiv 2507.18071, Qwen team): replaces token-level importance ratios with sequence-level (geometric-mean normalized) ratios + sequence-level clipping. Claimed fix for MoE RL collapse (routing instability makes token ratios noisy) and tolerance to inference/training engine precision mismatch. Used for Qwen3. **Real change**: granularity of the IS ratio; dense small models benefit less than MoE.
- **CISPO** (MiniMax-M1 tech report, ~2506, exact ID uncertain): clips the importance-sampling weights themselves rather than the update — keeps all tokens in the loss. Reported to matter mainly for stable long training at scale.
- **RLOO** (Ahmadian et al., 2024, "Back to Basics," 2402.14740) and **ReMax**: leave-one-out / greedy-baseline REINFORCE; RLOO ≈ GRPO without std normalization. REINFORCE++ (Hu et al., 2501.03262) adds PPO-style clipping + global reward normalization, no critic.
- **VinePPO / VC-PPO** (2409-ish): value-guided credit assignment to fix token-level credit in GRPO-style methods.
- **Lite-PPO, SRPO, SAPO, BAPO, Kimi k1.5's variant, OpenPipe/ART variants**: mostly hyperparameter/aggregation re-weights; marginal deltas over DAPO-class baselines. Consensus forming that **within-family differences ≈ 1–3 AIME points; data curation and step count matter more**.

**Honest assessment:** post-DAPO, algorithmic novelty has largely been (a) variance/stability patches, (b) IS-ratio granularity, (c) exploiting zero-gradient groups. No GRPO-successor has shown a qualitative capability jump at fixed compute.

---

## 2. The pass@k / "does RL add capability?" debate — the central controversy

**Skeptic side:**
- **"Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?"** (Yue et al., arXiv 2504.13837; NeurIPS 2025): base models overtake RLVR models at large k (up to ~1024); RL model's solvable set ≈ subset of base's. RL sharpens, doesn't expand. RL responses sit in base model's low-perplexity region. Distillation, unlike RL, *does* raise pass@k.
- **"The Invisible Leash"** (arXiv 2507.14843): support-constrained optimization; token entropy can rise while *answer-level* entropy falls — net diversity shrinks.

**Believer side:**
- **CoT-Pass@K** paper ("RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs," arXiv 2506.14245, MSR; ICLR 2026): pass@k gives false credit for right-answer/wrong-reasoning (short-answer guessing); requiring correct CoT, RLVR holds the lead at all k. Attacks the metric, not the phenomenon.
- **ProRL** (NVIDIA, arXiv 2505.24864; NeurIPS 2025): >2k RL steps with KL control + periodic reference reset + diverse tasks produces strategies the base can't reach even under heavy sampling — **on a 1.5B model** (Nemotron-Research-Reasoning-Qwen-1.5B, weights public). Boundary expansion correlates with training duration and base-model task competence. This is the single most relevant counter-result for small-model work.
- **"The Surprising Effectiveness of Negative Reinforcement"** (arXiv 2506.01347): decomposes RLVR into PSR/NSR; **negative-sample-only training improves pass@k across all k** (suppressing wrong mass redistributes to plausible candidates); PSR-only improves pass@1 but shrinks diversity. Suggests the boundary shrinkage is attributable to the *positive* gradient term.

**Where the ceiling actually is (working consensus):** RLVR reliably converts pass@256 capability into pass@1 capability; whether it expands the support depends on (a) training length, (b) base-model prior mass near the boundary, (c) KL/entropy management. With short runs and tight KL, the leash holds. Spurious Rewards (arXiv 2506.10947) adds the **model-family caveat**: random rewards gave +21 pts on Qwen2.5-Math-7B (amplifying latent "code-reasoning" prior) but failed on Llama/OLMo — many published Qwen gains overstate what RL contributes vs. what pretraining deposited.

---

## 3. Failure modes (well-characterized)

- **Entropy collapse**: "The Entropy Mechanism of RL for Reasoning" (arXiv 2505.22617) establishes empirical law R = −a·exp(H) + b — performance is spent entropy, ceiling predictable at H→0. Mechanism: covariance between token probability and advantage-driven logit change stays positive → monotone entropy decay. Fixes: Clip-Cov, KL-Cov (merged into verl). Related: Clip-Higher (DAPO), entropy bonuses (largely ineffective per multiple reports), forked-token entropy analyses.
- **Zero-advantage / sparse groups**: all-correct or all-wrong groups → no gradient. DAPO's dynamic sampling; **LENS** (arXiv 2510.08696) assigns confidence-weighted negative rewards to all-wrong groups; SGPO uses step-wise judge to diversify within groups (OpenReview).
- **Negative-gradient pathology**: "Lazy Likelihood Displacement" (arXiv 2505.18830) — uniform penalization of wrong-response tokens drags down *correct* response likelihood too (DPO-analogous). NTHR downweights; tested 0.5B–3B.
- **Length dynamics**: GRPO length bias inflates wrong answers (2503.20783); separate "overthinking"/length-bloat literature (e.g., "Demystifying Long CoT," O1-Pruner-style work); conversely length collapse under penalties.
- **Reward hacking / sycophancy to verifier**: format-gaming, answer-shortcutting, reward-model overoptimization (classic Gao et al. scaling laws still the reference).
- **Rollout diversity decay**: distinct-answer count drops with training even when token entropy rises (2507.14843); connects to pass@k ceiling.
- **Instability at scale**: irreversible collapse on MoE (GSPO motivation), train/inference mismatch (defective-importance-sampling analyses, e.g. "Your Efficient RL Framework Secretly Brings You Off-Policy RL" ~2508).

---

## 4. Emerging directions 2025–2026

- **Test-time / label-free RL**: **TTRL** (arXiv 2504.16084; NeurIPS 2025) — majority-vote-as-reward on unlabeled test inputs; +211% pass@1 AIME on Qwen2.5-Math-7B, exceeds initial maj@N ceiling, tested 1.5B–32B. Follow-ups: EVOL-RL, online TTRL variants.
- **RL without external rewards (RLIF)**: **INTUITOR** (arXiv 2505.19590; ICLR 2026) — self-certainty (KL-to-uniform) as the *only* reward; matches GRPO in-domain, generalizes better OOD. Related: entropy/rentropy-minimization fine-tuning ("Rentarn" work, EMPO "entropy-minimized policy optimization" ~2507?), self-rewarding consistency objectives. Active, contested area — known risk: confidence-hacking (model becomes confident on garbage).
- **Process rewards**: PRM800K lineage → Math-Shepherd → implicit PRMs (Yuan et al., "Free PRM"); PRMs as within-group judges for negative groups; still expensive/noisy at small scale.
- **Agentic/tool-use RL**: Search-R1, ReSearch, ToRL, R1-Searcher; multi-turn agentic RL (WebGPT-lineage → SWE-RL, DeepResearch-style). 2025–26 growth area; tool-call reward shaping largely open.
- **Continual/online RL**: ProRL's reference-reset trick; continual-RL catastrophic-forgetting studies sparse for LLMs — a genuine gap.
- **Experience replay for LLM RL**: largely absent from literature — on-policy dogma; a few works revisit off-policy correction (e.g., mixing replayed high-value traces, "RL with hindsight/HER-for-LLM" attempts, e.g., RLOO with buffer in agent settings). **Underexplored.**
- **World-model / lookahead RL for LLMs**: tree-search-based RL (TSO, AlphaLLM-style, V-Star), VPO; mostly inference-time, thin on training-time world models.

---

## 5. Small-model-specific findings

- ProRL's flagship demo is **1.5B** — boundary expansion shown at the smallest end (2505.24864).
- Dr. GRPO minimalist recipe: 7B, 27h on 8×A100, 43.3% AIME — SOTA-per-compute (2503.20783).
- **TinyZero** (Pan et al., 2025 replication of R1-Zero, ~$30): 3B and even 0.5B Qwen show self-verification emergence on countdown/multiplication — but gains mostly on **synthetic templated tasks**; generalization to natural math weak at ≤3B.
- NTHR validated 0.5B–3B (2505.18830); LENS on Qwen2.5-3B & Llama-8B (2510.08696).
- Community findings (OpenRLHF/verl/TinyZero replications): RL on **distilled** small models (e.g., R1-distill) yields small further gains; RL on **base** small models with narrow domains works; RL on small **instruct** models often collapses fast (entropy floor reached quickly).
- RL-vs-SFT headroom at small scale: consensus anecdote + several papers (e.g., "SFT memorizes, RL generalizes" Chu et al., 2501.17161) — RL gives better OOD generalization, SFT better at format/knowledge injection; at <3B, RL's absolute gains shrink because the base prior has thin mass on hard problems (the leash is shorter — consistent with 2504.13837).

---

## GAPS — what practitioners admit is unsolved (esp. small models / low compute)

- **The exploration leash**: no principled method to seed probability mass into *absent* solution regions; all fixes (clip-higher, entropy bonuses, cov-clip) only slow the shrink. Explicitly stated open problem in 2507.14843, 2504.13837.
- **Entropy is a consumable resource** (R = −a·e^H + b): nobody has a method to *replenish* entropy mid-run without breaking convergence; ProRL's reference-reset is the crude hack.
- **Zero-signal groups**: ~30–60% of rollouts at small scale are all-wrong/all-right → wasted compute. LENS/SGPO/dynamic-sampling are patches; curriculum sampling (difficulty-matched prompt selection) is under-formalized.
- **Off-policy/replay**: rollouts dominate cost (~70–80% of wall time); experience reuse is nearly untouched because IS corrections are unstable for sequences. Whoever cracks stable replay at 1–7B wins on compute.
- **Credit assignment**: still effectively outcome-only at scale; PRMs don't pay off below ~7B; token-level credit (VinePPO) too expensive for low-compute.
- **Model-family confound**: most results are Qwen-only; spurious-reward findings (2506.10947) show Qwen's pretraining does much of the work. Small-model claims need Llama/OLMo/Gemma validation to be credible.
- **Dense rewards without verifiers**: intrinsic signals (self-certainty, entropy-min) hackable; no robust intrinsic reward for open-ended tasks.
- **Small-base prior poverty**: at ≤3B the base pass@k support is thin — RLVR may literally have nothing to sharpen; combining tiny RL with tiny distillation or retrieval to *inject* mass before sharpening is a plausible novel axis (related: "distillation expands, RL sharpens" from 2504.13837).
- **Continual RL**: catastrophic forgetting + reference drift when running RL continually; essentially no mature recipe.
- **Evaluation honesty**: pass@k vs CoT-Pass@k dispute unresolved; benchmark contamination on Qwen widely suspected; small-model AIME numbers are high-variance.
- **Length/reasoning tradeoff**: no accepted principled method for token efficiency without accuracy loss at small scale (Dr. GRPO fixes bias, not optimal length).

**Most defensible novelty axes at 0.5B–14B / low compute** (survey synthesis): (1) replay/off-policy RLVR with stable sequence-level IS (GSPO's machinery is a natural substrate); (2) prior-injection + sharpening two-phase schemes (explicitly expanding support before RL); (3) negative-group exploitation (LENS-lineage is young); (4) entropy-replenishing curricula; (5) cross-family-validated intrinsic rewards. Claims should be benchmarked against DAPO/Dr.GRPO baselines on non-Qwen bases to dodge the two known confounds.

*Several citations (CISPO exact ID, EMPO, some follow-up TTRL variants) are "title approx" — verify before citing in a paper.*
