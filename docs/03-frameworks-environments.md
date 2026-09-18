# RL-for-LLMs Landscape Catalog (as of early 2026)

*Star counts are approximate — verified where search returned GitHub metadata; others are order-of-magnitude estimates from memory and should be double-checked. Dates reflect latest visible release notes.*

---

## 1. TRAINING FRAMEWORKS

| Framework | Org | ~Stars | Design idea / status |
|---|---|---|---|
| **verl** | ByteDance (Volcano Engine) | ~20k+ | The de-facto standard. HybridFlow paper. Hybrid-controller programming model; FSDP/Megatron trainers + vLLM/SGLang rollout; colocated (default) or `separate_async` disaggregated; AgentLoop for multi-turn; staleness-aware decoupled PPO, partial rollout. Very active. |
| **TRL** | HuggingFace | ~15k+ | Simplest entry point (GRPO/PPO/DPO trainers, HF-native). Experimental local async GRPO. Best for small models/single-node. Active. |
| **OpenRLHF** | community | ~8k | Early popular RLHF lib; DeepSpeed + Ray + vLLM; reward-model-centric RLHF. Active but less cutting-edge than verl/slime. |
| **slime** | Z.ai (Zhipu) + THUDM | ~3k | Megatron + SGLang, fully decoupled async (RPC), MoE-focused (trained GLM). Simple, easy custom data-gen hooks. Active. |
| **ROLL** | Alibaba | ~3k | "RL for agentic + reasoning + RLHF"; flexible/researcher-friendly, production-scalable. Active. |
| **AReaL** | Ant Research (inclusionAI) | ~3k | IMPALA/A3C-style fully async; stream rollout, staleness-aware decoupled PPO with η bound, heterogeneous GPU pools. Active. |
| **NeMo-RL** | NVIDIA | ~1–2k | NVIDIA stack; async agentic training; DTensor/Megatron. Active. |
| **SkyRL** (skyrl-train + SkyRL-Agent) | Berkeley/NovaSky | ~2k | Switchable colocated↔disaggregated; SkyRL-Agent adds async pipeline dispatcher (1.55× speedup) + tool-enhanced training; SA-SWE-32B result. Active. |
| **prime-rl** + **verifiers** | Prime Intellect | ~3k combined | Env-first ecosystem: `verifiers` = env/eval library + Environments Hub; prime-rl = async GRPO trainer; SHARDCAST/TOPLOC for untrusted cross-provider training (INTELLECT-2). Very active (verifiers v0.1.12, Apr 2026). |
| **Agent-Lightning** | Microsoft | ~17.9k (verified) | v1.0 rewrite: ~3.5k LoC; trains *any* agent harness with zero code changes via API-gateway proxy + LightningRL credit assignment; K8s-native rollout; runs verl+vLLM under the hood. Very active. |
| **ART (openpipe-art)** | OpenPipe | ~10.6k (verified) | Ergonomic GRPO wrapper; client runs anywhere, ephemeral GPU server; W&B Training integration; LoRA-friendly → hobbyist-accessible (Colab 2048 demo). Active. |
| **rLLM** | rllm-project | ~2k | Agentic RL framework; wraps verifiers envs; verl or Tinker backends. Active. |
| **Atropos** | Nous Research | ~1k | "LLM RL gym" — env collection + rollout management that plugs into trainers. Semi-active. |
| **UnstableBaselines** | TextArena team | <1k | Lightweight async online RL lib for TextArena games. Niche, active. |
| **RAGEN** | community | ~1k | Multi-turn agentic RL training (built on verl), used for StarPO work. |
| **MARTI / RLHFlow / NiuRL** | RLHFlow community | ~1k | Multi-agent LLM RL; semi-maintained, smaller community. |
| **mbridge** | ByteDance | <1k | Megatron↔HF weight bridge — infrastructure, not a full trainer. |
| **Tinker** | Thinking Machines | closed API | LoRA-RL-as-a-service; rLLM/SkyRL support it as a backend. |
| **exo / RL Swarm (Gensyn)** | exo labs / Gensyn | — | Decentralized/collaborative RL over distributed compute. |
| **trlx** (CarperAI) | archived/dead | — | Confirmed unmaintained; superseded by TRL. |
| **RL4LMs** (AllenAI) | stale | — | Gym-style GRPO4LMs; largely inactive since ~2023. |
| **ColossalAI-RL** | HPC-AI Tech | — | RLHF extension of ColossalAI; low activity. |

**Shared abstractions:** trainer (FSDP/Megatron/DeepSpeed) + rollout engine (vLLM/SGLang, increasingly as a *service* over HTTP/RPC) + weight-sync mechanism (NCCL broadcast, CUDA IPC, or checkpoint-via-filesystem) + async/staleness machinery (decoupled PPO objective, staleness threshold, partial rollout) + an env interface converging on either Gym-style `reset/step` (GEM, TextArena) or the `verifiers` env/rubric format.

---

## 2. ENVIRONMENTS & TASKSETS

| Env | What it provides | Reward type |
|---|---|---|
| **reasoning-gym** (open-thought) | 100+ procedural generators: algebra, arithmetic, graph, logic, games (Countdown, Rubik's); infinite data, controllable difficulty | Rule-based algorithmic verifiers |
| **verifiers Environments Hub** (Prime Intellect) | Community hub of envs incl. RLMEnv, opencode harnesses; envs = dataset + rubric + rollout logic | Rubrics: rule-based + LLM-judge + code exec |
| **TextArena** | 100+ text games, single/two/multi-player; SPIRAL self-play RL; MindGames (theory-of-mind) | Win/loss game outcomes, self-play |
| **GEM** (Axon RL) | OpenAI-Gym analogue for LLMs; 24 envs, async vectorized exec, wrappers; REINFORCE-ReBN baseline | Dense per-turn rewards possible |
| **KORGym** (ByteDance, M-A-P) | 50+ text/visual games, multi-turn interactive eval + RL scenarios | Rule-based game verifiers |
| **SWE envs**: SWE-Gym, R2E-Gym, SWE-bench-Verified harnesses, Agent-Lightning coding example | Issue→patch agent loops in containers | Test-pass execution rewards |
| **τ-bench / τ²-Bench / VitaBench / AppWorld / ToolSandbox** | Tool-use + simulated-user multi-turn tasks | DB-state checking + LLM judge |
| **Web/OS agent envs**: WebArena, OSWorld, WorkArena, AgentBench | Browser/desktop agents | Programmatic state checks |
| **Math suites as envs**: DAPO-Math-17k, DeepScaleR, OpenR1-Math | Fixed math datasets w/ answer extraction | Symbolic/exact-match verification |
| **Search envs**: Search-R1, deep-research tasksets | Retrieval tool loops | Answer F1/EM |
| **TinyZero/Countdown** | Minimal verifiable game | Rule verifier |
| **RandomWorld** (EMNLP'25) | Procedural generation of *tools themselves* + compositional tool-use DAGs | Synthetic, verifiable goal states |

**Reward taxonomy observed:** (a) rule-based verifiers (dominant for math/games); (b) code-execution/test outcomes (SWE, tool-use); (c) LLM-as-judge rubrics (open-ended, multi-turn); (d) majority-vote / pseudo-labels (weakly-supervised domains); (e) environment-level advantage (AutoForge's ERPO); (f) self-play outcomes.

---

## 3. PATTERNS WORTH NOTING

- **Auto environment generation is an emerging subfield:** AutoForge (arXiv 2512.22857 — synthesizes envs+tasks from tool docs via tool-dependency-graph random walks + env-level RL); RandomWorld (procedural tools); OMNI-EPIC and Eurekaverse (LLM-written envs+reward fns, mostly robotics/sim); ATLAS (co-designs tasks+levels as reward machines).
- **Procedural infinite tasks:** reasoning-gym is the flagship; KORGym and RandomWorld follow.
- **Difficulty scheduling/curriculum:** reasoning-gym difficulty params; MILA "Self-Evolving Curriculum"; UED lineage (PAIRED→ATLAS).
- **Self-play as reward source:** SPIRAL on TextArena — zero-sum self-play incentivizing reasoning.
- **Unusual reward domains:** RL on compression (RLMEnv context-management tasks), retrieval/search, SQL, computer-use, memory agents (SkyRL-Agent case studies).
- **Decentralized/untrusted RL:** Gensyn RL Swarm, INTELLECT-2 (TOPLOC inference proofs).
- **Env engineering admitted-hard:** AutoForge explicitly cites simulated-user instability and env heterogeneity; Agent-Lightning ships reward-hacking-prevention scripts; SkyRL's dispatcher exists because stragglers/long-horizon rollouts dominate cost. Contamination is the stated motivation for procedural envs (reasoning-gym, KORGym both argue fixed benchmarks are memorized).

---

## 4. WHITESPACE — underexplored, cheap-enough niches

1. **Procedural tool/env generation for small models specifically.** RandomWorld/AutoForge exist but target general agents; nobody ships a "tool-gym generator" tuned for 1–8B models with difficulty calibrated to small-model success rates. Cheap (pure CPU envs).
2. **Dense per-turn reward shaping for tiny models.** GEM argues REINFORCE-ReBN > GRPO for dense rewards, but almost all envs give sparse terminal rewards — a dense-reward env suite for small models is open.
3. **Reward-hacking-robust env design as a first-class artifact.** Envs that deliberately include hackable shortcuts and measure/train-against exploitation — barely explored outside adversarial examples.
4. **Compression/memory/context-management tasks.** RLMEnv is nearly alone. Small models on constrained context are the perfect fit; procedural "context tetris" tasks are cheap to build.
5. **Self-improving env generators (env-gen agent trained by RL).** AutoForge/OMNI-EPIC generate envs but the *generator itself* isn't RL-trained on student learning-progress signals (UED-style regret) for LLM tasks — the classic UED loop hasn't been ported to text envs at hobbyist scale.
6. **Multi-agent / theory-of-mind training envs beyond games.** MindGames competition is eval-focused; cheap multi-turn negotiation/coordination tasksets with verifiable outcomes for small models are thin.
7. **Negative-result / abstention training envs.** Envs where reward is for correctly refusing unsolvable instances — procedurally generatable, rarely built.
8. **Compute-asymmetry curricula:** envs that reward *efficient* reasoning (token/turn budgets as part of reward) tuned for small models — partial precedents (OptimalThinkingBench is eval-only).
9. **Env hubs lack ratings/difficulty telemetry.** The Prime Intellect Hub has no standardized difficulty calibration or per-model-size leaderboards — a "curated small-model env ladder" doesn't exist.
10. **Retro-tool RL:** training small models to be *good tools inside bigger agents' harnesses* (sub-agent reward = parent task success) — Agent-Lightning's plumbing supports it but no env/taskset targets it.

**Caveats:** Star counts for everything except Agent-Lightning (~17.9k) and ART (~10.6k) are estimates. trlx/RL4LMs/ColossalAI-RL status inferred from repo activity; verify on GitHub before citing.
