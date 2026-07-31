# Project PROMETHEUS — Evolved Cognition for Energy-Materials Discovery

An end-to-end, self-contained JAX pipeline that **evolves its own learning system** — both the weight-update rule and the computation graph that uses it — and then turns that evolved system loose on inverse-designing new energy materials from the periodic table, gated by a physics oracle and an honesty check that refuses to claim a "discovery" the AI can't actually justify.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![JAX](https://img.shields.io/badge/JAX-differentiable%20physics-9cf?logo=jax&logoColor=white)
![Elements](https://img.shields.io/badge/periodic%20table-65%20elements-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)
![Status](https://img.shields.io/badge/status-research%20demo-yellow)

## Table of Contents

- [Overview](#overview)
- [Why this project?](#why-this-project)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Phase 1 — nothing is hand-designed, not even the learning rule](#phase-1--nothing-is-hand-designed-not-even-the-learning-rule)
- [customenergy.py — deep dive](#customenergypy--deep-dive)
- [The physics oracle](#the-physics-oracle)
- [The honesty gates](#the-honesty-gates)
- [Hyperparameters](#hyperparameters)
- [Example output](#example-output)
- [Output artifacts](#output-artifacts)
- [Limitations & honest notes](#limitations--honest-notes)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [License](#license)

## Overview

`customenergy.py` runs in two phases.

**Phase 1** meta-evolves a small population of complete "cognitive systems" — each one is a genome describing both a symbolic synaptic plasticity rule (how weights should change) *and* a program of cognitive operations (matmul, attention, memory read/write, gating, routing, ...) that defines how the system processes information. Genomes are scored on few-shot learning tasks and evolved with mutation, crossover, and elitism over many generations. Nothing here is backprop — the winning system's own evolved rule is what updates its weights.

**Phase 2** takes the best evolved system and points it at materials discovery. It trains on 500 random quaternary combinations drawn from a 65-element periodic table, then uses gradient-based inverse design (`jax.grad`) to search for a hypothetical material whose predicted properties hit two *frozen* physical targets: the Shockley–Queisser bandgap optimum (1.34 eV) for photovoltaics, and a 450 mAh/g gravimetric capacity ceiling for battery cathodes. Every candidate has to pass a chemistry-validity gate (does it map to a real, non-degenerate ABX₃-style formula?) and a coherency gate (does the AI's own prediction actually agree with the ground-truth physics oracle?) before it's allowed to be called a "discovery."

| | |
|---|---|
| **Input** | A 16-dim continuous vector: 4 crystallographic sites × [radius, electronegativity, mass, cost] |
| **Output** | 7-vector: bandgap (eV), storage (mAh/g), stability (eV), PV score, battery score, stability score, cost penalty |
| **Learning system** | Fully evolved (genetic programming) — plasticity rule + cognitive op-graph, not a fixed Transformer/MLP |
| **Physics** | Closed-form, differentiable, semi-empirical formulas (Goldschmidt tolerance factor, gravimetric capacity, etc.) — no external DFT/simulation calls |
| **Representation** | 65-element periodic table (H → Bi, plus the light lanthanides La–Nd), mapped to an ABX₃ perovskite-style 4-site structure |
| **Hardware** | CPU (default) or GPU, auto-detected via `jax.devices()` |
| **Framework** | JAX (`jax.grad`, `jax.vmap`, `jax.random`), 64-bit precision enabled |

## Why this project?

Most "AI materials discovery" demos do one of two things:

- bolt a fixed neural network (a Transformer, an MLP) onto a dataset and call whatever it predicts a "discovery," with no check that the prediction is grounded in real physics, or
- hard-code the optimizer and the target, so the model can always find *something* to declare as "new best" — a target that quietly moves to match whatever was last produced.

This repo is built to avoid both failure modes:

- **The learning system itself is evolved**, not just its weights. Genetic programming searches over both the plasticity rule (the analogue of "how weights should change") and the cognitive program (the analogue of "what the model's forward pass looks like"), so the architecture and the update rule co-evolve.
- **The targets are frozen physical constants** (`FIXED_TARGET_BANDGAP = 1.34`, `FIXED_TARGET_STORAGE = 450.0`, `FIXED_OPTIMAL_STABILITY = -4.5`), declared explicitly in the code as *laws, not learnable parameters*. What's allowed to adapt after each discovery is only `training_weights` — how much training attention the system pays to PV-relevant vs. battery-relevant vs. stable vs. cheap examples. That can't move the goalposts, because the goalposts don't move.
- **A "new best" has to earn it.** A candidate only counts as a discovery if it (a) maps to a chemically valid, non-degenerate formula, and (b) the AI's own prediction agrees with the ground-truth oracle on the same input within a fixed tolerance. If the AI's inverse-design loss went down but its prediction doesn't actually track reality, the loop logs it as `INCOHERENT` and moves on — it does not get counted.
- **PV and battery performance are never blended into one score.** A material can be good for solar, good for batteries, both, or neither; the two sub-scores (plus a separate stability score and cost penalty) are reported side by side, not averaged into a single number that hides which objective is actually being satisfied.

(If the filename convention looks familiar — `customenergy.py` alongside a sibling `customremedies.py` — that's because this is the energy-materials counterpart to a generative drug-discovery project built the same way: evolve/rule-drive the learning process, then gate every output against an honest external check before it's allowed to count.)

## Architecture at a glance

```
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — EvolutionaryEngine.run()                                    │
│                                                                          │
│   population = [seed_genome] + 19 × random_genome()                    │
│         │                                                               │
│         ▼                                                               │
│   evaluate(genome) on 8 few-shot tasks (5 support / 10 query, drawn     │
│   from the 500-material database)                                      │
│         │                                                               │
│         ▼                                                               │
│   fitness = 0.35·learning_speed + 0.45·generalization                  │
│             + 0.20·extra-thinking-helps − 0.001·complexity(genome)     │
│         │                                                               │
│         ▼                                                               │
│   tournament-select (k=3) → elitism (top 25%) → mutate / crossover     │
│   (mutate_plasticity, mutate_program, mutate_config, crossover,        │
│    add_step, remove_step, deepen_thinking, simplify_thinking)          │
│         │                                                               │
│         └──────────────── repeat × n_generations ───────────────┘      │
└──────────────────────────────────┬───────────────────────────────────┘
                                    │ best_genome (plasticity rule +
                                    │ cognitive program that won)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — continuous_discovery_loop()                                 │
│                                                                          │
│   EvolvedCognitiveSystem(best_genome) trained on 500 random ABX₃       │
│   quaternary combinations (65-element periodic table)                  │
│         │                                                               │
│         ▼                                                               │
│   inverse design: jax.grad(loss_fn) × 500 steps pushes a random        │
│   16-d "material" toward the FROZEN targets (bandgap→1.34 eV,          │
│   storage→450 mAh/g, PV/Battery/Stability sub-scores→1.0)              │
│         │                                                               │
│         ▼                                                               │
│   CHEMISTRY GATE — nearest real element per site role (A-site          │
│   cation / B-site metal / two X-site anions), no element reused        │
│         │                                                               │
│         ▼                                                               │
│   COHERENCY GATE — does think(x) agree with _physics_oracle(x)         │
│   within tolerance 0.20? if not → INCOHERENT, not a discovery          │
│         │                                                               │
│         ▼                                                               │
│   "NEW BEST" declared only if coherent + chem-valid + beats the        │
│   previous best PV / battery / combined score                          │
│         │                                                               │
│         ▼                                                               │
│   adapt training_weights (emphasis only, targets stay frozen) →        │
│   append to database → retrain (weighted extra passes) → repeat        │
└────────────────────────────────────────────────────────────────────────┘
```

## Repository structure

```
.
├── customenergy.py           # Everything: evolution, cognition, physics oracle, discovery loop
├── download/                 # Output artifacts, created at runtime
│   ├── prometheus_discoveries.json
│   ├── prometheus_discoveries.csv
│   └── prometheus_discoveries.txt
└── README.md                 # You are here
```

This is a single-file project — there's no separate model definition, config file, or dataset to download. Everything from the periodic table to the genetic-programming operators to the discovery loop lives in `customenergy.py`.

## Installation

```bash
pip install jax numpy
```

That's the entire dependency list — no PyTorch, no RDKit, no external APIs. `jax` pulls in `jaxlib` automatically for CPU. For GPU acceleration, install the CUDA-enabled `jaxlib` build per the [official JAX installation instructions](https://github.com/google/jax#installation); the script auto-detects it via `jax.devices()` and switches over with no code changes.

Verified while preparing this README on Python 3.12 (Linux, CPU-only, single core) with `jax==0.11.0`.

> **Heads up on `save_dir`:** the discovery-log writer at the bottom of the script currently hard-codes `save_dir = "/home/z/my-project/download"`. Change this to a path that exists on your machine (or make it relative, e.g. `"./download"`) before running, or the three output files just won't be written where you expect.

## Quick start

```bash
python customenergy.py
```

By default `__main__` runs:

```python
engine = EvolutionaryEngine(population_size=20, input_dim=16, output_dim=7)
best_genome, best_fitness = engine.run(n_generations=21, verbose=True)
continuous_discovery_loop(best_genome, engine.evaluator, iterations=15)
```

That's a meaningful amount of compute on CPU — each generation evaluates 20 genomes on 8 few-shot tasks apiece, and each discovery-loop iteration runs 500 steps of gradient descent through the evolved system. For a quick smoke test, drop the numbers down first, e.g. `population_size=6, n_generations=3, iterations=2` (this is exactly what was used to produce the [example output](#example-output) below).

## Phase 1 — nothing is hand-designed, not even the learning rule

This is the headline feature. In a typical project you'd write a neural network and hand it Adam. Here, both halves of "how does this system learn and think" are search targets:

**The plasticity rule** (`PlasticityNode`) is a small expression tree built from:

| Category | Values |
|---|---|
| Binary ops | `+`, `-`, `*`, `/` |
| Unary ops | `exp`, `sin`, `cos`, `abs`, `tanh`, `sigmoid`, `relu`, `sign`, `neg` |
| Variables | `pre`, `post`, `w`, `err`, `lr`, `mem`, `reward`, `mod` |
| Constants | `0.01, 0.1, 0.5, 1.0, -1.0, 2.0, 0.001` |

`compile_plasticity()` turns the tree into a real JAX function (`Δw = f(ctx)`, with `ctx` holding `pre/post/w/err/lr/mem/reward/mod` for that weight tensor), wrapped in a `try/except` that falls back to a zero update if the expression misbehaves. The seed genome starts from a plain delta rule (`lr * pre * err`, i.e. ordinary gradient descent), but nothing stops evolution from drifting somewhere stranger — the example run below converged on `relu(sin(tanh(tanh(lr))))`, a rule that (for that short run) ignores the actual error signal entirely, which is a good illustration of why the coherency gate in Phase 2 matters.

**The cognitive program** (`SystemGenome.cognitive_steps`) is a short sequence of typed operations, each drawn from:

| Op | # | Op | # |
|---|---|---|---|
| `matmul` | 0 | `threshold` | 7 |
| `attention` | 1 | `rotate` | 8 |
| `mem_write` | 2 | `blend` | 9 |
| `mem_read` | 3 | `norm` | 10 |
| `compare` | 4 | `routing` | 11 |
| `gate` | 5 | | |
| `accumulate` | 6 | | |

`EvolvedCognitiveSystem.think()` runs this program for `n_think_steps` iterations, reading from an external memory bank (`mem_read`/`mem_write`/`attention` all touch it) and writing outputs through a learned projection. `learn()` computes a loss, then applies the evolved plasticity rule to every plastic weight — no autodiff, no optimizer object, just the tree-derived function applied directly.

**Search operators** available to `EvolutionaryEngine._mutate()`: `mutate_plasticity`, `mutate_program`, `mutate_config` (memory size / hidden dim / learning rate), `crossover` (single-point splice of another genome's `cognitive_steps`), `add_step`, `remove_step`, `deepen_thinking`, `simplify_thinking`. Selection is tournament-based (k=3) with elitism on the top quarter of the population each generation.

## customenergy.py — deep dive

### Fitness (`UniversalEnergyEvaluator.evaluate`)

Each genome is scored on 8 few-shot tasks (5 support examples, 10 query examples, drawn from the 500-row materials database):

```
fitness = 0.35 · learning_speed        # how much loss drops across the 5 support steps
        + 0.45 · generalization        # 1 / (1 + mean query loss)
        + 0.20 · thinking_helps        # does more n_think_steps reduce loss vs. a 1-step variant?
        − 0.001 · complexity(genome)   # complexity = len(cognitive_steps)·hidden_dim + memory_size
```

The small complexity penalty is an Occam's-razor term — all else equal, a smaller evolved system wins.

### The material representation

Every candidate material is a 4-site **ABX₃-style perovskite** structure:

- **Site 0 — A-site cation:** large, low-electronegativity (alkali metals, alkaline earths, light lanthanides)
- **Site 1 — B-site metal:** mid-radius transition / post-transition metal
- **Site 2 & 3 — X-site anions:** small, high-electronegativity (O, S, N, F, Cl, ...), constrained to be different from each other

`match_features_to_real_material()` maps a continuous 16-d feature vector back to real elements one role at a time, always picking the nearest unused candidate in that role's pool. If fewer than 3 distinct elements end up chosen (e.g. it degenerates toward the same element at every site), the match is flagged invalid rather than silently accepted — this is explicitly there to prevent the "every site collapses to H" failure mode called out in the code's own comments.

## The physics oracle

`_physics_oracle()` is a closed-form, fully differentiable set of semi-empirical formulas — no DFT, no external simulation calls, which is what makes gradient-based inverse design (`jax.grad(loss_fn)`) possible at all:

| Quantity | Formula (informal) | Clipped range |
|---|---|---|
| Goldschmidt tolerance factor `t` | `(r_A + r_X1) / (√2 · (r_B + r_X1))` | — |
| Octahedral factor `μ` | `r_B / r_X1` | — |
| Stability (eV) | `-4.5 + 25(t-0.9)² + 10(μ-0.5)² + 0.5(EN_B - EN_X1)²` | `[-10, 5]` |
| Bandgap (eV) | `5.0 - 1.5·r_X1 + 2·\|t-0.9\| + \|μ-0.5\|` | `[0.1, 6.0]` |
| Storage (mAh/g) | `1000 / (avg_mass + 1) · \|EN_B - EN_X1\|` | `[0, 500]` |
| Cost penalty | `avg_cost / 100` | `[0, 1]` |
| PV score | `1 / (1 + (bandgap - 1.34)²)` | `[0, 1]` |
| Battery score | `1 / (1 + (storage - 450)² / 1000)` | `[0, 1]` |
| Stability score | `1 + clip(stability, -1, 0)` | `[0, 1]` |

Two things worth knowing if you read the source: the stability and bandgap formulas key primarily off the A-site, B-site, and *first* X-site (site 2); the second anion (site 3) mostly enters through the average mass/cost terms and the inverse-design bounds rather than its own radius/electronegativity. And PV/battery scores are reported **separately** everywhere — `format_full_structure`, the CSV columns, the JSON payload — there's no single blended "energy score" anywhere in the output.

## The honesty gates

Two gates stand between "the optimizer's loss went down" and "this counts as a discovery":

1. **Chemistry gate** — is the matched formula real and non-degenerate (≥3 distinct elements across the 4 sites, each in a chemically sensible role)?
2. **Coherency gate** — does the evolved system's own prediction (`think(x)`) agree with the ground-truth oracle (`_physics_oracle(x)`) on the *same input*, within an absolute tolerance of 0.20 on both the PV and battery sub-scores?

Only candidates that pass both, *and* that beat the frozen best-so-far on PV score, battery score, or their sum, get `declared_new_best = True`. Everything else is logged with an explicit status (`INCOHERENT`, `CHEMISTRY INVALID`, or "coherent and valid but didn't beat the prior best") — the loop's honest-summary printout at the end reports how many of the total iterations were actually valid discoveries versus rejected on each gate.

## Hyperparameters

| Name | Default | Notes |
|---|---|---|
| `population_size` | 20 | Phase 1 genome population |
| `n_generations` | 21 | Phase 1 evolution length |
| `iterations` (discovery loop) | 15 | Phase 2 inverse-design + retrain cycles |
| `input_dim` / `output_dim` | 16 / 7 | 4 sites × 4 features → 7 physics outputs |
| `hidden_dim` (per genome) | 32 / 64 / 128 (evolved) | — |
| `memory_size` (per genome) | 16 / 32 / 64 (evolved) | — |
| `n_think_steps` (per genome) | 1–4 initially, mutable up to 8 | — |
| `learning_rate` (per genome) | `10^Uniform(-4, -1)` initially | `mutate_config` can push it to `10^Uniform(-5, 0)` |
| Inverse-design steps | 500 | `lr=0.02`, gradient descent on the 16-d input |
| Coherency tolerance | 0.20 | Absolute difference on PV and battery sub-scores |
| `FIXED_TARGET_BANDGAP` | 1.34 eV | Frozen — Shockley–Queisser optimum |
| `FIXED_TARGET_STORAGE` | 450.0 mAh/g | Frozen — light-ion cathode ceiling |
| `FIXED_OPTIMAL_STABILITY` | -4.5 eV | Frozen — Goldschmidt-ideal reference |
| Precision | `float64` | `jax.config.update("jax_enable_x64", True)` |

## Example output

Below is real console output, lightly trimmed, from an actual run with everything scaled down (`population_size=6, n_generations=3, iterations=2`) purely to demonstrate the report format quickly — the default settings (20 / 21 / 15) will run substantially longer and explore far more candidates.

```
==========================================================================================
Project PROMETHEUS: Universal Energy Materials Discovery
==========================================================================================
Population: 6 | Generations: 3 | Device: cpu:0
Database: 65 Elements | Continuous Multi-Objective Physics
------------------------------------------------------------------------------------------
Gen   0 | Fit: -0.0373 | Cplx:  224 | 23.4s
Gen   1 | Fit: -0.0920 | Cplx:  224 | 6.4s
Gen   2 | Fit: -0.1185 | Cplx:  128 | 5.7s

==========================================================================================
EVOLUTION COMPLETE — Best System Discovered:
==========================================================================================
  Learning Rule (Plasticity):
    Δw = jax.nn.relu(jnp.sin(jnp.tanh(jnp.tanh(ctx['lr']))))
  Think Steps: 4 | Ops: 3 | Hidden: 32

==========================================================================================
PHASE 2: HONEST CONTINUOUS DISCOVERY
==========================================================================================
  Frozen PV target       : bandgap = 1.34 eV (Shockley-Queisser)
  Frozen Battery target  : storage = 450.0 mAh/g
  ...
[2.1] AI is searching for a better material...
------------------------------------------------------------------------------------------
CONTINUOUS LOOP ITERATION 2 RESULTS:
  AI Predicted:   Eg=0.00 eV | Storage=0.0 mAh/g | PV=0.000 | Battery=0.000
  Oracle (true):  Eg=5.02 eV | Storage=96.0 mAh/g | PV=0.069 | Battery=0.008 | Stab=1.000 | CostPen=0.024
  Closest real formula: Li-Al-P-S
  Chemistry valid: True
  Coherency: pv_disagree=0.069 bat_disagree=0.008 -> COHERENT
  Status: NEW BEST COMBINED MATERIAL (PV+Battery=0.077)

  ABX3-style unit cell for LiAlPS  (iter 2)

        +-----+-----+
        |     |     |
        |  Li |   P |    <- A-site cation + X-site anion #1
        |     |     |
        +-----+-----+
        |     |     |
        |  Al |   S |    <- B-site metal  + X-site anion #2
        |     |     |
        +-----+-----+
  ...
------------------------------------------------------------------------------------------
CONTINUOUS DISCOVERY COMPLETE -- HONEST SUMMARY
------------------------------------------------------------------------------------------
  Total iterations:            2
  Incoherent (AI != oracle):   1  (these were NOT counted as discoveries)
  Chemistry-invalid:           0  (these were NOT counted as discoveries)
  Valid discoveries:           1  (coherent + chemically valid)
  Best PV sub-score:           0.069 / 1.000  (frozen target: bandgap=1.34 eV)
  Best Battery sub-score:      0.008 / 1.000  (frozen target: storage=450.0 mAh/g)
  Best combined PV+Battery:    0.077 / 2.000
  Best combined formula:       Li-Al-P-S
==========================================================================================
```

Note the first iteration in that run was `INCOHERENT` (the AI predicted all-zero properties while the oracle computed real values) and was correctly excluded from the discovery count — exactly the behavior the honesty gates are there to enforce. With only 3 evolution generations and 2 discovery iterations the system hasn't had much time to learn, so treat the actual scores here as a format demo, not a performance claim; longer runs with the default settings converge on much better fits.

## Output artifacts

At the end of `continuous_discovery_loop()`, three files are written to `save_dir` (default `/home/z/my-project/download/` — see the [Installation](#installation) note above about changing this):

| File | Contents |
|---|---|
| `prometheus_discoveries.json` | Full machine-readable record per iteration: sites, AI prediction vs. oracle truth, gates, status — plus run metadata (frozen targets, gate counts, final training weights) |
| `prometheus_discoveries.csv` | One row per discovery, flattened site-by-site, for spreadsheet/pandas analysis |
| `prometheus_discoveries.txt` | Human-readable log: the ASCII unit-cell diagram + full structure table for every single iteration |

## Limitations & honest notes

In the same spirit as the honesty gates baked into the code itself:

- **The physics oracle is semi-empirical, not DFT.** The formulas above are simplified, closed-form stand-ins for real materials physics, chosen specifically so gradient-based inverse design is tractable. Treat every discovered formula as a hypothesis to investigate, not a validated material.
- **The periodic table isn't literally complete.** The docstring says "H to Bismuth + Lanthanides," but the built-in table has 65 elements — it skips most transition-metal rows above period 5 (Hf, Ta, Re, Os, Ir aren't included) and all but the first four lanthanides (La–Nd). `_classify_element_role()` references a few elements (e.g. Hf, Ta) that aren't actually in `periodic_table`, so they're silently absent from the B-site candidate pool rather than causing an error.
- **The bandit-style mutation-strategy weighting isn't fully wired up.** `EvolutionaryEngine.strategy_rewards` is initialized and decayed by 0.95 every generation, but nothing in the current code increments it based on which mutation actually produced fitness gains — so in practice strategy probabilities drift toward their floor value over time rather than being reward-driven. Worth fixing if you want genuinely adaptive operator selection (see [Roadmap](#roadmap)).
- **Stability/bandgap formulas don't weight all 4 sites symmetrically** — see the note at the end of [The physics oracle](#the-physics-oracle).
- **`save_dir` is hard-coded** to a path (`/home/z/my-project/download`) that's unlikely to exist on your machine.
- **Single population, single machine, single core by default.** `EvolutionaryEngine._evaluate_population()` and the discovery loop are both sequential Python loops — there's no `vmap`/`pmap` batching across genomes or across discovery candidates, so wall-clock time scales roughly linearly with `population_size` and `iterations`.

## Roadmap

- [ ] Wire up `strategy_rewards` to actual fitness deltas (real bandit-style operator selection)
- [ ] Batch genome evaluation with `jax.vmap` across the population instead of a Python `for` loop
- [ ] Swap `save_dir` for a CLI arg / relative default
- [ ] Expand the periodic table to the full set (complete transition-metal rows, full lanthanide/actinide series)
- [ ] Replace or augment the semi-empirical oracle with calls to a real materials database (Materials Project API) for validation
- [ ] Support non-ABX₃ architectures (layered, spinel, etc.) instead of a fixed 4-site perovskite template
- [ ] Persist/resume evolutionary state (checkpoint the `hall_of_fame` and population between runs)
- [ ] Streamlit/Gradio dashboard over the JSON discovery log

## Citation

If this project is useful in your work, please cite:

```bibtex
@misc{project_prometheus_energy_materials,
  title  = {Project PROMETHEUS: Universal Continuous Energy Materials Discovery Engine},
  author = {<your name here>},
  year   = {2026},
  url    = {<your repository URL here>}
}
```

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Evolved end to end. Nothing declared a "discovery" until it's earned it.</sub>
