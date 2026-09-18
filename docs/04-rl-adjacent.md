# RL-LIKE-BUT-NOT-RL: A Map of the Adjacent Possible for Small-Model Post-Training

*Survey of paradigms that deliver RL-style improvement through non-vanilla-RL machinery, plus adjacent-field RL variants not yet (or barely) tried on LLMs. Arxiv IDs given where confident; flagged as approximations where not. "Cheap" = plausibly runnable on a few GPUs at 1–7B scale.*

---

## 1. ITERATED SELF-IMPROVEMENT LOOPS (best-of-n → distill → repeat)

The canonical "EM-flavored RL": sample, filter by a checker/reward, SFT on survivors, repeat. Structurally this is policy iteration with the improvement step replaced by rejection sampling + supervised learning.

| System | What it showed | Small-scale? | Cheap? |
|---|---|---|---|
| **STaR** (Zelikman et al., 2203.14465) | Bootstrap rationales: sample CoT, keep correct ones, "rationalize" failures by hinting the answer. GPT-J 6B → large gains on GSM8K/CommonsenseQA. | Yes (6B) | Yes |
| **ReST** (Gulcehre et al., 2308.08998) | Growing-batch RL framing: Grow (sample) / Improve (offline RL on filtered data). Machine translation; more compute-efficient than online RLHF. | Yes | Yes |
| **ReST-EM** (Singh et al., 2312.06585) | Formal EM view; PaLM-2 on MATH/APPS. **Key finding: saturates after ~2 iterations; overfits small problem sets.** Scales with model size. | Partially (PaLM-2 S) | Yes |
| **RAFT** (Dong et al., 2304.06767) | Best-of-n + SFT, one or few iterations. Simple, widely used baseline. | Yes | Very cheap |
| **Iterative RFT / Online RFT** (Yuan et al., 2305.20045) | Rejection sampling FT repeated; diversity of the *problem set* is the bottleneck. | Yes (7B) | Yes |
| **B-STaR** (2412.17256) | Diagnoses saturation: self-improvement stalls after 3–5 iters; balances exploration/exploitation via dynamic config. Mistral-7B, Llama3-8B. | Yes | Yes |
| **Self-Rewarding LM** (Yuan et al., 2401.10020) | Model is its own judge (LLM-as-judge reward) + Iterative DPO. Llama2-70B; AlpacaEval gains, ~3 iters useful. | No (70B) | Moderate |
| **SPIN** (Chen et al., 2401.01335) | Self-play: model must distinguish its own generations from SFT gold data — a GAN-like discriminant game reducing to iterative DPO. Zephyr-7B. | Yes (7B) | Yes |
| **Self-Instruct / bootstrapping** (Wang et al., 2212.10560) | Instruction data self-generation; quality-filtered. Mostly a data factory, not iterative weight improvement. | Yes | Yes |
| **Iterative DPO / online DPO** (e.g., Xiong et al., "Gibbs sampling from human feedback"; Snorkel DPO-iter works) | Regenerate preference pairs each round against a fixed or self reference; ~3–4 rounds help, then plateau. | Yes | Moderate |
| **Self-play preference opt (SPPO)** (Wu et al., ~2405) | Game-theoretic self-play where policy tries to beat its own previous versions via preference loss. | Yes (small) | Yes |

**Where they saturate / open problems:** verified across B-STaR and ReST-EM: 3–5 iterations then plateau. Causes: (a) fixed problem set exhaustion, (b) diversity collapse (entropy shrink), (c) no mechanism to *expand* the frontier — filtering can only select from what the model already reaches pass@k. The interesting frontier: **adversarial curriculum generation** (make problems harder as you improve — see §6) and **on-policy error-correction data** (STaR's rationalization trick is underexplored).

---

## 2. SEARCH-GUIDED / INFERENCE-TIME LEARNING

| System | What it showed | Small-scale? | Cheap? |
|---|---|---|---|
| **rStar-Math** (2501.04519) | MCTS rollouts generate step-verified trajectories; trains policy SLM + process preference model (PPM) via 4 rounds of self-evolution. Qwen2.5-Math-7B: 58.8→90.0 MATH; Phi3-mini-3.8B: 41.4→86.4; AIME 53.3%. **The strongest existence proof that search-then-distill beats scale.** | Yes (1.5–7B) | Moderate (MCTS rollouts are the cost; still far cheaper than training a 70B) |
| **rStar** (2408.06195) | Mutual-consistency MCTS decoding at test time, no training. | Yes | Cheap at inference |
| **AlphaLLM / AlphaZero-like tree search for LLM** (~2404.12247) | MCTS + value model distilled into the policy; math reasoning. | Yes | Moderate |
| **TextGrad** (2406.07496) | "Verbal gradients": LLM critiques backpropagate through a text computation graph to optimize prompts/code. | Yes | Cheap |
| **DSPy / MIPRO / OPRO** (DSPy 2310.03714; OPRO 2309.03409) | Compile-time optimization of prompts/few-shots via search + LLM-as-optimizer. Not weight learning, but a cheap capability multiplier that could feed distillation. | Yes | Cheap |
| **Reflexion** (Shinn et al., 2303.11366) | Verbal self-critique stored in episodic memory; agent-level improvement without weights. | Yes | Cheap |
| **FunSearch** (DeepMind, Nature 2023) | Evolutionary search over *programs* with LLM as mutator + evaluator as fitness; solved cap-set. | Yes (works with small LLMs as mutators) | Cheap |
| **AlphaEvolve** (DeepMind, 2025) | FunSearch scaled: evolutionary code improvement, matrix-mult kernels, data-center scheduling. Uses Gemini but the loop is model-agnostic. | Untested at 7B | Moderate |
| **EvoPrompt / Promptbreeder / PhaseEvo** | Evolutionary/self-referential prompt populations. Promptbreeder: prompt mutates its own mutation-prompt — closest to "self-improving search operator". | Yes | Cheap |
| **Tree-search distillation, general** (e.g., ToT→distill works; "Distilling System 2 into System 1", 2407.06023) | Sample expensive search, filter, SFT — amortize inference cost into weights. Works for branch-and-bound, lookahead, CoT. | Yes | Yes |

**Key insight for the small-model use case:** search-then-distill is the most proven "RL-like" factory at small scale (rStar-Math). Underexplored variants: distill the *search operator itself* (the value/reward model as a co-evolving SLM — rStar-Math does this, few others do), and **evolutionary program synthesis as a general reward-free domain** (execution = free verifier).

---

## 3. EVOLUTIONARY / GRADIENT-FREE WEIGHT METHODS

| System | What it showed | Small-scale? | Cheap? |
|---|---|---|---|
| **MeZO** (Malladi et al., 2305.17333) | ZO-SGD via SPSA; forward passes only, inference-memory fine-tuning, matches Adam within ~1% on many tasks; **can optimize non-differentiable objectives (accuracy, F1, reward-model scores)**. OPT up to 66B; works great with LoRA. | Yes | Very cheap memory; many forward passes though |
| **LOZO** (ICLR 2025) | Low-rank ZO estimator exploiting low-rank gradient structure; beats MeZO. | Yes | Yes |
| **RoZO** (EACL 2026) | Riemannian ZO on the LoRA manifold (parallel transport, trust regions). | Yes | Yes |
| **LOREN** (AAAI 2026) | Natural evolution strategies + low-rank curvature preconditioner + RLOO estimator; −27% memory vs MeZO-Adam. | Yes | Yes |
| **Black-Box Tuning / BBTv2** (Sun et al., 2201.03514 / ~2210) | CMA-ES over a low-dim subspace of prompt/prefix params. API-only tuning. | Yes | Yes |
| **OpenAI-ES-style direct weight evolution on LLMs** — essentially **untried at scale**; only small results exist (e.g., EvoLLM-type work, "finetuning LLMs with ES" scattered 2024–25 workshop papers). NES theory says effective dim ~ LoRA rank makes this feasible. | — | — |
| **Sakana evolutionary model merge** (2403.13187, Nature Machine Intelligence) | CMA-ES over merge recipes in param-space AND data-flow-space; produced SOTA Japanese-math LLM with zero training. | Yes (7B-scale merges) | Very cheap (evals only) |
| **M2N2** (Sakana, 2508.16204) | Evolutionary merging with dynamic split-points + competition-for-data diversity + attraction pairing; evolved MNIST classifiers from scratch rivaling CMA-ES. | Yes | Moderate |
| **CycleQD** (Sakana, ICLR 2025) | **MAP-Elites/QD applied to model weights**: merging = crossover, SVD = mutation, QD selection over skill-niche archives → population of diverse 8B agents. The clearest QD-for-LLM-weights result. | Yes (8B) | Moderate |
| **Population-based training (PBT) for LLMs** — used for hyperparams at big labs (unpublished); open results thin. | — | — |

**Gap:** ZO methods have been aimed at *SFT-equivalent* objectives. Using MeZO/ES directly on **non-differentiable rewards (execution results, verifier scores, debate wins) on a LoRA** is almost unexplored — a literal "RL factory without gradients."

---

## 4. REWARD-FREE / LABEL-FREE SIGNALS

| System | Signal | What it showed |
|---|---|---|
| **TTRL** (Zuo et al., NeurIPS 2025; ~2504.16084) | Majority vote over own rollouts as pseudo-label → GRPO/PPO reward | Qwen-2.5-Math-7B +~211% relative on AIME 2024, unlabeled test data only; **can exceed its own maj@n ceiling**. Tested 1.5–7B, moderately cheap. |
| **DARE** (2601.21804) | Full rollout distribution (not just majority) + exploration bonus + pruning | +25% rel. AIME over MV-TTRL |
| **SCOPE** (ACL 2026) | Step-wise confidence-weighted pseudo-labels + subgroup partitioning for diversity | +13% rel. AIME'25 |
| **TTRL-CoCoV** (2606.03608) | Confidence-conditioned: verify only medium-confidence samples; exploration reward for high-conf | +9.8% pass@1, +18.7% pass@16 over TTRL |
| **CoRE** (2608.09324) | Replicator-dynamics consensus over rollout graph (agreement + similarity + confidence) | Beats MV-TTRL; reaches plateau in 54–70% fewer steps |
| **Self-certainty / confidence-as-reward** (e.g., "RLSC" ~2506, "confidence reward" works 2025) | Reward = model's own answer confidence; no labels, no voting | Modest gains; risk: rewards overconfidence |
| **Entropy objectives** (e.g., "entropy bonus / min-entropy RL", EMPO ~2504) | Optimize or shape token-entropy | Entropy collapse is now recognized as *the* RLVR failure mode; entropy-as-signal underexplored |
| **INTUITOR** (2505.19590) | Intrinsic reward = KL between model's own conditional distributions (self-certainty-like), no external reward | Matches RLVR on GSM8K with zero labels |
| **Execution-free verifiers / consistency rewards** (e.g., SCPO self-consistency preference opt; cross-checking code-vs-text answers) | Symmetry/consistency checks | Works for math/code-adjacent tasks; cheap |
| **Curiosity/intrinsic motivation for LLMs** | — | **Almost untried.** Classic ICM/RND have no mainstream LLM analog; nearest is entropy bonuses and novelty-scored problem generation |
| **Active inference / energy-based rerankers** | Free-energy minimization | Only toy/positional papers for LLMs; EBM rerankers explored pre-2023, largely abandoned after RLHF |

**Frontier here:** the TTRL cluster proves *self-consistency is a trainable reward*. The untried step is combining consensus-reward RL with **expanding difficulty** (self-generated problems whose difficulty is calibrated so consensus is informative) — that becomes an autocurriculum.

---

## 5. EXOTIC TRANSFERS (RL subfields barely touched for LLMs)

| Technique | LLM status |
|---|---|
| **Successor features / generalized policy improvement** | **Untried.** No serious LLM work factorizing reward into features×weights for rapid task transfer. |
| **Options / hierarchical RL** | Partially: "options" appear as tool/subagent calls; skill-token methods (e.g., "CoT options", hierarchical RLVR with subgoal rewards) exist in a few 2024–25 papers but nothing canonical. Open. |
| **HER (hindsight experience replay)** | Analogs exist implicitly: STaR's "rationalization" (relabel failed attempt with the answer) *is* HER. Explicit goal-relabeling for LLM agents — one paper ("LLM-Driven Relabeling", AAMAS'26) does HER→LLM in the robotics direction, not the reverse. **LLM-HER for agentic tasks is open.** |
| **Offline RL / Decision Transformer for post-training** | LaMo (ICLR'24) shows LM-initialized DTs beat offline RL on MuJoCo. Reverse direction — return-conditioned or advantage-conditioned *generation* as post-training — only partially explored (e.g., "Steering LLMs via reward-conditioning", exogenous-reward prompting works, Q-Transformer-adjacent). **Largely open.** |
| **World-model rollouts / dreaming** | MuZero-style imagined rollouts for LLMs: only simulation-of-environments works (WebArena world models, "Simulacra"-type); no Dreamer-for-text policy improvement. Open. |
| **Regret minimization / CFR** | Counterfactual regret = essentially what DPO's regret-bounded analysis gestures at (DPO derived as bounded-regret); explicit CFR over text games — a couple of poker/negotiation papers (e.g., "Language Model CFR", ReBeL-adjacent). Mostly open. |
| **Bandit feedback theory** | Used heavily in prompt-selection/A-B-routing; contextual-bandit fine-tuning formalized (e.g., "RLHF as contextual bandit" analyses, VCBPPO). Underused as a *training* signal. |
| **Distributional RL** | **Untried.** No C51/QR-DQN analog for LLM value/reward heads; distributional reward models could enable risk-aware generation. Open. |
| **Curiosity / empowerment** | Empowerment (maximize mutual info between actions and outcomes) has *no* LLM implementation; curiosity only as novelty-scoring. Open. |
| **Homeostatic / energy-based learning** | Equilibrium propagation, predictive coding — exist for small nets; **no LLM-scale demo**. PC-based transformer training papers exist at toy scale. Open. |

---

## 6. MULTI-AGENT / SOCIAL

| System | What it showed |
|---|---|
| **Debate as training signal** | Mostly *evaluation* (Du et al. multi-agent debate improves answers at inference; Michael et al. debate for scalable oversight). Training-on-debate-outcome is thin: a few 2025 works distill debate winners. **Open as a reward factory.** |
| **Adversarial task generation / PAIRED-style UED** | PAIRED (Dennis et al.) never directly ported; nearest: AgentGym-style curriculum, "self-generated curriculum" works (e.g., adversarial math problem generators, "Task Me Anything"), and **Absolute Zero / AZR** (2505.03335) — self-play proposer/solver with execution feedback, no external data, strong math+code gains at 7B. **The single most relevant system to this direction.** |
| **Self-play beyond games** | SPIN/SPPO (§1) are self-play in disguise; AlphaZero-style generator-verifier self-play = AZR. Self-play between *specialized roles* (attacker/solver/judge) mostly open. |
| **Population diversity maintenance** | CycleQD/M2N2 (§3) on weights; on *outputs*, diversity-maintenance RL is the entropy-collapse literature (§4); QD over behavior descriptors of LLM outputs is **open**. |
| **Voyager-style skill libraries** | Skill accumulation in Minecraft; distilling discovered skills into small-model weights underexplored. |
| **SWE-RL-style dual-role self-play** (Meta, 2025) | Agent plays bug-injector and bug-fixer — adversarial co-evolution on real tasks; early but directly relevant. |

---

## UNEXPLORED CROSSINGS — adjacent-field machinery apparently NOT applied to LLM post-training

1. **ES/zoRL on non-differentiable rewards.** MeZO proved forward-only fine-tuning works on LoRA and can optimize any scalar. Point it at verifier/debate/execution rewards = RL with zero backprop. Nobody has published this combination at 7B scale.
2. **MAP-Elites over *behavior descriptors* of generations** (not weights): archive cells = (answer style, length, reasoning-pattern cluster); select diverse elites for SFT — a diversity-preserving alternative to best-of-n that may resist the 3–5-iteration saturation of ReST-EM.
3. **Successor features for multi-task RLHF:** factor reward = w·φ(response); instant policy transfer to new reward weights without retraining.
4. **Explicit HER for agentic LLMs:** relabel failed trajectories with achieved goals → train on "what you actually did." Massive cheap-data factory for agent tasks; only hinted by STaR rationalization.
5. **Distributional reward models / risk-sensitive decoding:** C51-style value distributions enable CVaR-pessimistic or optimistic sampling — unexplored.
6. **Empowerment / option-discovery objectives:** intrinsic reward = maximize mutual information between latent skills and outcome tokens; would give unsupervised *skill discovery* for hierarchical small models.
7. **Decision-Transformer-style conditioning as post-training:** train the model return/quality-conditioned at SFT time, then prompt "quality=high" — free controllable capability dial; only partially done.
8. **CFR / counterfactual regret over self-play populations** (debate, prover-solver): formal no-regret dynamics instead of ad-hoc iterative DPO.
9. **QD for problem generation (MAP-Elites curriculum):** archive of problems indexed by difficulty/features; evolve problems that sit at the model's "consensus boundary" (where majority vote is ~50/50 — maximally informative for TTRL-style rewards). Couples pseudo-rewards with PAIRED-style UED — nobody has done it.
10. **World-model "dreaming" for agentic tasks:** learn a text world model (cheap — it's just the LLM on env transcripts), roll out imagined trajectories, train policy on them. Dreamer-for-text is open.
11. **Predictive-coding / equilibrium-propagation fine-tuning:** local learning rules → fine-tune without global backprop; at 1–7B scale this is speculative but genuinely absent from literature.
12. **Coevolutionary archives (POET-style):** co-evolve a population of *environments/problem-generators* alongside solvers; Absolute Zero does generator-solver self-play but without a maintained open-ended archive of niches — POET machinery is the missing piece.
13. **Bandit/EXP3-style mixture-of-experts training:** treat LoRA experts as arms, regret-minimizing allocation — formal guarantees, untried.

### Proven-cheap foundations to build on
- rStar-Math (2501.04519): search→distill self-evolution, 7B beating o1-preview.
- TTRL + successors (2504.16084, DARE, CoRE): label-free consensus rewards that exceed their own supervision ceiling.
- Absolute Zero (2505.03335): self-play curriculum with execution rewards, zero external data.
- MeZO lineage (2305.17333, LOZO, LOREN): gradient-free weight updates at inference memory.
- Sakana suite (2403.13187, CycleQD, M2N2): QD/evolution on weights, cheap.

### Flags / approximations
- Some 2025–26 arxiv IDs are approximate (SPPO, INTUITOR, RLSC, AlphaEvolve) — verify before citing.
- "Untried" claims mean: no significant published result found as of this survey; workshop/preprint exceptions may exist, especially for #1, #4, #8.
- Findings rely on search snippets + prior knowledge, not full-paper reads.
