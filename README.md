# aitraindisrupt

Research notes + a catalog of novel post-training ideas: "factories of reinforcement
learning" — mechanisms that look like RL but aren't vanilla RL — aimed at pushing
**small models (0.5B–14B)** past their known ceilings on **restrained compute**.

Produced 2026-09-18. Grounded in a sweep of the 2024–2026 RL-for-LLM literature,
the GLM-5.x industrial case study, the GitHub frameworks/environments landscape,
and RL-adjacent paradigms (evolutionary methods, self-improvement loops, label-free
rewards, exotic RL subfields). Full digests in [`docs/`](docs/).

---

## What the walls actually are

- **The leash**: RLVR converts pass@256 capability into pass@1 but mostly can't
  expand the support (Yue et al., 2504.13837). Counter-evidence: ProRL expanded the
  boundary at 1.5B with >2k steps + reference resets (2505.24864); GLM-5.3 pushed
  the *same base* as 5.2 purely by scaling environments. The wall is real but
  breachable via duration + env supply.
- **Entropy is a consumable**: R = −a·e^H + b (2505.22617). Everyone has tricks to
  *slow* entropy burn (Clip-Higher, cov-clip); **nobody has a way to refill it**
  mid-run. Open problem.
- **Rollouts are 70–80% of cost** and experience replay is nearly untouched because
  sequence-level IS correction is unstable. Whoever cracks stable replay wins on
  compute.
- **30–60% of rollouts are zero-signal** (all-right/all-wrong groups give no gradient).
- **Self-improvement loops saturate in 3–5 iterations** (ReST-EM, B-STaR) — filtering
  can only pick from what the model already reaches.
- **Environments are the new bottleneck** — Zhipu's own thesis post-5.3: "the
  difficulty moved from the model to the environment."
- **Continual RL forgets** — GLM-5 treats on-policy cross-stage distillation as
  mandatory between stages.
- **The Qwen confound**: many published small-model gains are pretraining deposits,
  not RL (Spurious Rewards, 2506.10947). Validate on Llama/Gemma/OLMo.

## The idea catalog

Scores = novelty estimate vs. published literature as of the sweep (1–10). Nothing
scored 9+ on purpose — true 9s would be speculative rather than engineerable.

### A. Breaking the pass@k leash (inject + sharpen, not just sharpen)

| Name | Novelty | One-liner |
|---|---|---|
| **PriorGraft** | 7 | Before RL, cheaply micro-distill *only* the boundary problems (near-misses) so probability mass exists where the model is blind — then RL has something to sharpen. |
| **EntropyBank** | 8 | Mid-run, pause RL and do a tiny SFT pass on a diversity archive of past solutions to refill the entropy you've spent, then resume. |
| **EdgeFeed** | 8 | Auto-generate problems calibrated so the model's majority-vote is ~50/50 — that's where label-free consensus rewards (TTRL) carry maximum information. |
| **SkillMine** | 8.5 | Reward the model for maximizing mutual information between a latent "skill" it chose and the outcome — unsupervised skill discovery RL can later compose. |

### B. Recycling rollouts (the compute moat)

| Name | Novelty | One-liner |
|---|---|---|
| **TraceVault** | 7 | A replay buffer for LLM RL: reuse valuable old rollouts with GSPO-style sequence-level importance correction + staleness decay. Directly attacks the 70–80% rollout cost. |
| **HindsightLedger** | 8 | Failed an agentic trajectory? Relabel it with the goal it *actually* achieved and train on that (HER for text) — every failure becomes valid data. |
| **SleepCycle** | 7 | Wake: on-policy RL. Sleep: offline consolidation pass — replay, hindsight-relabel, distill the day's best/worst rollouts — then wake again. |
| **DreamRoll** | 8.5 | Train a cheap text "world model" on real env transcripts; the policy practices in imagination between real (expensive) env calls. |

### C. Reward/environment factories (the GLM-5.3 axis, at hobbyist scale)

| Name | Novelty | One-liner |
|---|---|---|
| **SymmetryMint** | 7 | Mint verifiable tasks from any corpus via invertible transforms (solve↔generate, code↔spec); reward = round-trip consistency. Free verifiers, infinite supply. |
| **HackGym** | 7.5 | Ship envs with deliberate reward shortcuts; reward the model for finding exploits, then patch the verifier — adversarial co-evolution of reward quality. |
| **RefusalMine** | 6.5 | Procedurally generate *unsolvable* instances and reward correct refusal — trains calibration, not just capability. |
| **TaskFoundry** | 6.5 | Pipeline that synthesizes executable, hack-tested environments whose difficulty auto-calibrates to a small model's measured success rate. |

### D. Loops that look like RL but aren't

| Name | Novelty | One-liner |
|---|---|---|
| **ZeroGradGym** | 8 | Optimize verifier rewards directly on a LoRA using forward-passes-only evolution (MeZO/LOREN-style) — RL with zero backprop, fits a 7B on one consumer GPU. |
| **ArchiveForge** | 8 | POET-style: maintain a growing archive of *problem-generator* niches co-evolving with the solver — Absolute Zero does self-play but keeps no archive. |
| **BehaviorMap** | 7.5 | Replace best-of-n with MAP-Elites over reasoning-behavior clusters — select *diverse* elites for SFT, which is what self-improvement loops lose when they plateau. |
| **RegretTable** | 7 | Run proposer/solver/judge populations under formal no-regret (CFR-style) dynamics instead of ad-hoc iterative DPO rounds. |
| **PessiMist** | 7 | Distributional reward head: sample *optimistically* during exploration, act *pessimistically* at eval — risk-aware RL, untried for LLMs. |
| **LoRACasino** | 6.5 | A bandit (EXP3) that allocates rollouts across a shelf of skill-specific LoRA adapters, provably minimizing regret per FLOP. |
| **LadderKeep** | 7 | Each RL stage distills into a fresh swappable LoRA "skill cartridge" instead of the base — skills never overwrite each other, so continual RL stops forgetting. |
| **ReturnDial** | 6 | SFT with quality-score tags, then at RL time condition on "quality=high" — a free capability dial baked in at training (Decision-Transformer trick). |

## If scoping week one (engineer's pick)

1. **TraceVault** — highest ROI in the list; stable replay is the known-unsolved
   cost bottleneck and GSPO's sequence-level machinery is a ready substrate.
2. **ZeroGradGym** — cheapest real experiment: one GPU, a LoRA, reasoning-gym
   verifiers, no training infra. If it moves pass@1 at all, it's a paper.
3. **EdgeFeed** — compounds with TTRL (proven) and attacks the leash head-on;
   needs only a problem generator + a vote-counting loop.
4. **HindsightLedger** — pairs with any agentic env and turns the biggest waste
   stream (failed trajectories) into a data factory.

## Honesty notes

- Novelty scores are relative to what the sweep found published; workshop
  preprints may exist for the 7–8 range.
- Deliberately **excluded**: another PPO variant (that soup is saturated —
  1–3 AIME points each), plain entropy bonuses (proven ineffective), PRMs for
  small models (too expensive per the field's own admission).
- Whatever gets built, benchmark on a **non-Qwen base** (Gemma/Llama/OLMo) —
  that's the credibility move half the field skips.

## Research digests

| Doc | Contents |
|---|---|
| [docs/01-rl-frontier.md](docs/01-rl-frontier.md) | 2024–2026 RL-for-LLM frontier: GRPO-variant lineage, the pass@k debate, failure modes, emerging directions, small-model findings, admitted gaps |
| [docs/02-glm-case-study.md](docs/02-glm-case-study.md) | How far RL alone pushed the GLM family (4.5 → 5.3), SAO, industry RL deltas, lessons for small-model RL |
| [docs/03-frameworks-environments.md](docs/03-frameworks-environments.md) | Training framework catalog (verl, TRL, slime, ART, Agent-Lightning…), env/taskset catalog, whitespace |
| [docs/04-rl-adjacent.md](docs/04-rl-adjacent.md) | RL-like-but-not-RL paradigms: self-improvement loops, search-then-distill, evolutionary weight methods, label-free rewards, untried crossings |
