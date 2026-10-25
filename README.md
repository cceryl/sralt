# SRALT

SRALT studies social-relation inference from black-box recommendation outputs and a small public social graph, focusing on long-tail users. It supports Ciao, Epinions, Flickr and Yelp through target, shadow and attack stages.

## Current Restoration Status

As of 2026-10-25, `target/model=design` restores SRALT's implementation from commit `b8b6854d6f1289b141cbd9764a2876a37e9df6d4`, retaining graph-cache lifecycle fixes and the shared data/evaluation protocol. Its revision is `legacy-restored-v1`. The later original-paper DESIGN, SocialPP DESIGN, and their DESIGN-specific SAL variants have been removed from the working tree; Git history and archived reports retain their provenance.

**Only DESIGN has been restored so far.** SMIA shadow and MLP still use the current author-aligned implementation, and GDMSR/EGCL have not been rolled back. Current pipeline outputs must not be described as the complete August baseline or compared as if they were the historical results.

Obsolete tuning, best10 and SAL-specific scripts and tests have been removed. Public sampling, state restoration, caching and metric fixes remain. See [the restoration task](docs/on-going/legacy-implementation/README.md), [cleanup record](docs/on-going/legacy-implementation/cleanup-record.md), and [TODO](todo.md).

## Installation

The project targets Linux and Python 3.12.13. Install with:

```bash
uv sync
source .venv/bin/activate
```

Use the project container for experiments. Current resource restrictions prohibit using the local GPU; experiment scheduling is maintained separately in TODO.

## Quick Start

Hydra requires explicit target data/model and shadow/attack choices. The generic pipeline remains available:

```bash
python scripts/pipeline.py target/data=ciao target/model=design shadow/model=smia attack/model=mlp
```

The low-memory entry releases each model after its last use and avoids extra run checkpoints, while retaining the model/data cache:

```bash
python scripts/pipeline_low_memory.py target/data=ciao target/model=design shadow/model=smia attack/model=mlp
```

These commands describe current capabilities, not a claim that the complete old chain has been restored or an instruction to start experiments now.

## Models

| Stage | Available models |
| --- | --- |
| Target | Restored DESIGN; current GDMSR, EGCL and non-social LightGCN; GDMSR/EGCL SAL subclasses |
| Shadow | SMIA, CSA, DSA, MINER adapter, TailProxy; optional no-shadow path |
| Attack | MLP and its existing variants, MINER/En_MINER, TailProxy, Rec-only and public-degree baselines |

For example, select `target/model=lightgcn` for a non-social target, or pair `shadow/model=miner` with `attack/model=miner` / `enminer`. Current SMIA, MLP and their budgets are documented in [models](docs/reference/models.md). Removed DESIGN variants are not active options.

## Epinions Data

Use `target/data=epinions`. The loader reads `data/epinions/rating.mat` and `trustnetwork.mat`. Epinions and Ciao share MATLAB preprocessing, filtering, global IDs and seeded random leave-one-out; the dataset protocol has not been rolled back.

Default preprocessing previously retained 18,061 users, 253,571 items and 713,231 interactions, with 571,306 directed social entries including 224 self-links. These are preprocessing records, not new results for the restored model.

## Reports and Outputs

`python scripts/comparison_report.py` produces the general comparison report. Explicit `--targets` and `--pairs shadow:attack` select available configurations. By default it does not train missing models, but can evaluate existing caches; this is not a read-only JSON listing. Use `--dry-run` for plan/cache inspection without evaluation. Historical metric cohorts must not be pooled with restored-model results.

Hydra outputs go to `runs/outputs/` or `runs/multirun/`; model/data caches remain in `.cache/`. The restored DESIGN configuration uses its new revision rather than reusing author-aligned weights. Do not delete all caches or rename old weight files into new identities.

## Documentation

- [Current task and TODO](todo.md)
- [Documentation index](docs/README.md)
- [Research protocol](docs/research-protocol.md)
- [Architecture and models](docs/reference/README.md)
- [Running and cache guide](docs/experiments-and-cache.md)
- [Completed studies](docs/archive/README.md) and [retired plans](docs/legacy/README.md)

Archived findings remain version-specific. This cleanup did not run tests, training or model evaluation.
