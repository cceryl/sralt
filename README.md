# SRALT

SRALT is a social-relation level membership inference attack framework targeting long-tail users in social recommender systems. It evaluates privacy leakage under strict black-box settings and is built around three stages: a target stage for training and evaluating the victim recommender, a shadow stage for learning attack-oriented user representations, and an attack stage for predicting whether a social link exists between two users.

The codebase supports Ciao, Epinions, Flickr, and Yelp datasets. The social target recommenders are DESIGN, GDMSR, and EGCL; LightGCN is available as a general (non-social) recommender baseline. The shadow models are SMIA, CSA, and DSA, and the final attack model is MLP. CSA, our proposed model, leverages residual vector quantization and semantic edge augmentation to learn high-quality representations for long-tail users. DSA, which removes the residual vector quantization module, serves as an ablation variant of CSA. SMIA serves as the baseline and employs two simple graph neural networks to learn user representations. The attack model is an MLP classifier that takes the representations of two users as input and predicts whether a social link exists between them.

MINER and En_MINER are also available as social-relation attack baselines, following the [SMIA authors' implementation](https://github.com/XuHao-bit/SMIA/tree/a7e79411244a574ba0b539510f105094a8956682). They use a dedicated shadow adapter and classify distance features between recommended item embeddings and another user's embedding.

SMIA now follows the authors' row-normalized propagation, regularization and raw-table attack features. The shared MLP uses unweighted BCE and clipped SGD without BatchNorm. Public social positives are split before shadow training; every shadow model receives only the attack training positives. Recommendation holdout, training budgets, validation selection and reporting retain SRALT's protocol. See the [alignment notes](docs/archive/smia-author-alignment.md).

DESIGN follows its original supplementary implementation in PyTorch, including batch edge removal, degree-dependent self weights, independent social/interaction chains, pointwise BCE and mutual distillation. It retains SRALT's shared data, training and evaluation protocol. Its `author-aligned-v1` configuration separates new model caches from the earlier implementation; see the [DESIGN migration notes](docs/archive/design-reproduction.md).

## Project Structure

- `docs/`: current protocol and running guide, with reference material in `docs/archive/` and dated history in `docs/legacy/`; start with [docs/README.md](docs/README.md)
- [todo.md](todo.md): current status and TODOs by protocol, target models, attacks (shadow design, attack design, baselines), defenses, and experiments; includes priorities, commands, and implementation prerequisites
- `scripts/`: scripts for end-to-end training and evaluation
- `sralt/`: package source code
- `sralt/configs/`: Hydra configuration files
- `data/`: dataset files
- `runs/`: Hydra outputs and experiment results
- `.cache/`: local cache for generated intermediate artifacts to speed up subsequent runs

GDMSR and EGCL follow the authors’ released model logic within SRALT’s shared protocol. Their configurations include `revision: author-aligned-v1` to separate corrected model caches from earlier implementations. See [the reproduction and cache migration notes](docs/archive/target-reproduction.md).

## Installation

The repository targets Linux and uses Python 3.12.13.

Make sure `uv` is installed first, then run:

```bash
uv sync
```

This creates a Python 3.12.13 virtual environment, fetches the required packages, and installs the project in editable mode.

You may need to activate the virtual environment before running the pipeline:

```bash
source .venv/bin/activate
```

## Quick Start

The pipeline is configured with Hydra. For a single run, you must at least specify `target/data`, `target/model`, `shadow/model`, and `attack/model` explicitly. A runnable example is:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=csa attack/model=mlp
```

To release each model after its last use and skip extra data/model checkpoints in
`runs`, use `scripts/pipeline_low_memory.py` with the same arguments (including
`-m` and experiment presets). It retains the existing `.cache` behavior, evaluation,
logs, Hydra configuration and TensorBoard output:

```bash
python ./scripts/pipeline_low_memory.py target/data=ciao target/model=design shadow/model=csa attack/model=mlp
```

To use the non-social LightGCN target, select `target/model=lightgcn`:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=lightgcn shadow/model=csa attack/model=mlp
```

LightGCN uses only user-item interactions. It follows the same data preprocessing, evaluation, and deployment-query protocol as the social targets. The existing comparison preset continues to select DESIGN, GDMSR, and EGCL.

All modifiable parameters are listed in the configuration files under `sralt/configs`. You can override any parameter by specifying its path and value on the command line. For example, to change the number of quantizers in the CSA shadow model, you can use the following command:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=csa attack/model=mlp shadow.model.attribute.num_quantizers=16
```

For a multirun sweep, add `-m` and override the values you want to explore. The following command sweeps over 4 datasets, 3 target models, 2 shadow models, and 5 random seeds, producing 120 sequential runs in total:

```bash
python ./scripts/pipeline.py -m target/data=ciao,epinions,flickr,yelp target/model=design,gdmsr,egcl shadow/model=smia,csa attack/model=mlp seed=42,43,44,45,46
```

Preset experiment configurations are stored in `sralt/configs/experiment`. They cover comparison, ablation, hyperparameter, and attacker-knowledge settings. For example, the following command runs the comparison experiment, which is equivalent to the previous multirun command:

```bash
python ./scripts/pipeline.py -m experiment=comparison
```

## Epinions Data

Select `target/data=epinions` in either pipeline, for example:

```bash
python scripts/pipeline_low_memory.py target/data=epinions target/model=design shadow/model=csa attack/model=mlp
```

The loader reads `data/epinions/rating.mat` (`rating`: user, item, category, rating) and `trustnetwork.mat` (`trustnetwork`: source user, destination user). Epinions and Ciao share MATLAB preprocessing: ratings >= 2, deduplication, undirected social edges, iterative filtering (at least 5 interactions and 1 social connection per user, 1 interaction per item), contiguous global IDs, and seeded random leave-one-out. Raw social self-links are retained, matching existing preprocessing.

The supplied Epinions files contain 922,267 rating rows and 355,754 trust rows. Default preprocessing retains 18,061 users, 253,571 items, 713,231 interactions, and 571,306 directed social entries (including 224 self-links). Train/validation/test interactions number 677,109 / 18,061 / 18,061. Seed 42 preprocessing and cache reload were verified on CPU; full training and parameter tuning remain pending. MINER/En_MINER use an initial Epinions learning rate of 0.05, not an author-reported optimum.

Both comparison sweeps and the comparison report include Epinions by default. Other ablation and parameter presets retain their Ciao default; override `target/data=epinions` to use the new dataset.

## EGCL Target-Only Tuning

Run `python scripts/tune_egcl.py` to train only EGCL, using Flickr, seeds 42–46,
learning rates 0.01/0.003/0.001, and early-stopping patience 50 (15 sequential trials).
Other model settings come from `sralt/configs/target/model/egcl.yaml`.

```bash
# Preview the grid without loading data or training.
python scripts/tune_egcl.py --dry-run
# Quick first pass on one seed.
python scripts/tune_egcl.py --seeds 42
# Replace the default grid; fixed overrides use the same model-relative keys.
python scripts/tune_egcl.py --datasets flickr ciao --seeds 42 \
  --grid attribute.weight_reg=60,20,6 attribute.embedding_reg=2,0.5 \
  --set train.optimizer.lr=0.003 --output runs/egcl_regularization
```

The script reuses the pipeline's target data/model caches. Rerunning a command
loads completed models and recomputes evaluation; an interrupted training trial
restarts from scratch. Each trial saves its target config, log and TensorBoard
events. `results.json` contains completed trials from the current invocation;
`summary.csv` / `summary.json` aggregate by dataset and model configuration,
ordered by validation Recall+NDCG. Compare equal seed sets (`n`/`seeds`), especially
while a sweep is incomplete. Use different output directories for separate sweeps.

Diagnostics use full-history deployment recommendations with history masking:
unique recommended items, item coverage, the fraction of all recommendation slots
occupied by the most frequent K items, mean user cosine similarity and first
singular-direction energy. K defaults to 30 (`--num-recommendations`); query
batch size defaults to 256 (`--query-batch-size`). These diagnose homogeneity;
validation recommendation metrics select models. `--no-diagnostics` skips this
extra evaluation; `--test-metrics` additionally reports test metrics without using
them for ranking. `--cpu` explicitly selects CPU. No shadow or attack model is
trained, and the existing EGCL YAML is not modified.

## Social MINER Baselines

Select the MINER shadow adapter together with the desired attack model:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=miner attack/model=miner
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=miner attack/model=enminer
```

MINER uses four distances (L1, L2, cosine, Bray-Curtis) from one user's K recommended items to the other user's embedding, giving `4K` attack features. En_MINER concatenates both directions, giving `8K` features. Both use the learned social embedding table and the SMIA authors' MLP architecture, unweighted binary cross entropy, and SGD with gradient clipping. Shadow training reuses SRALT's SMIA training settings; relation splits, negative sampling, validation selection, and All/Head/Tail metrics follow SRALT's protocol.

These reproduce the social baselines released with SMIA, which omit the original interaction-level MINER's knowledge graph, bilateral attack network, and cumulative learning. See [the reproduction notes](docs/archive/miner-reproduction.md) for sources and implementation differences.

The dedicated sweep covers both attack modes, four datasets, three social target models, and five seeds (120 sequential runs):

```bash
python ./scripts/pipeline.py -m experiment=miner
```

## Output Structure

Hydra writes single-run outputs to `runs/outputs/YYYY-MM-DD/HH-MM-SS` and multirun sweeps to `runs/multirun/YYYY-MM-DD/HH-MM-SS`. The pipeline also saves target, shadow, and attack data/model checkpoints inside each run directory.

## Interactive Comparison Report

Run `python scripts/comparison_report.py` to write `runs/comparison.html`, a single offline HTML file with filters, sortable tables, charts, seed averages/sample standard deviations, individual seed results, and CSV/JSON export. The page can hide combinations with no attack records across the selected seeds; partially missing combinations remain visible. Edit `DATASETS`, `TARGET_MODELS`, `MODEL_PAIRS`, `SEEDS`, and the display options at the top of the script. Dataset and target selections are crossed with the explicit shadow/attack pairs; this is independent of `experiment=comparison` and supports any compatible configured models. Default report targets are DESIGN, GDMSR, EGCL, and LightGCN; the training sweep preset is unchanged.

By default the script reads its metric cache first, then tests matching trained model caches when metrics are absent. If a required trained model is missing, the result is marked missing without training. To also train missing models, explicitly enable:

```bash
python scripts/comparison_report.py --allow-training
```

CLI options override the script defaults, for example:

```bash
python scripts/comparison_report.py --datasets ciao --targets design lightgcn \
  --pairs csa:mlp miner:miner miner:enminer --seeds 42 43 --allow-training
```

Attack tables report AUROC, F1, and TPR at 1% FPR for All/Head/Tail using the existing evaluation protocol. Select a Shadow/Attack pair in the page’s Baseline dropdown to show relative changes beside every attack metric: `(current - baseline) / baseline * 100%`. Comparisons match the dataset and target, and also the seed in individual-seed mode. Aggregated comparisons use each model’s mean over the selected available seeds, whose counts may differ. The reference remains available when its rows are hidden by Shadow/Attack filters. Missing or zero baseline values yield an undefined change (—). CSV export includes the chosen baseline and per-metric percentages; charts retain absolute values and include changes in their tooltips. Target/Shadow recommendation tests are enabled by default; control them independently with `--target-metrics` / `--no-target-metrics` and `--shadow-metrics` / `--no-shadow-metrics`, then hide or show them in the page. Use `--no-target-metrics`, `--no-shadow-metrics`, or `--no-allow-training` to override enabled script defaults. `--dry-run` lists selected runs and metric cache availability without evaluating or writing HTML. Relative output paths resolve from the repository root.

Only this script uses `.cache/comparison_report/`. Cache hits bypass model imports, loading, tensor fingerprinting, and testing. Keys include resolved stage dependencies, seeds, determinism, source/config contents, dataset file metadata, and `SRALT_CACHE_VERSION`. Target/Shadow metrics are shared across downstream model combinations. Same-configuration retraining or manual changes to training caches are not detected automatically: use `--refresh-metrics` to retest with existing model caches, without enabling training. `SRALT_CACHE_REFRESH=1` retains its pipeline meaning (including retraining) when training is allowed; it never enables training by itself. Existing pipeline runs do not populate this metric cache; the first report run tests their matching model caches automatically. `--dry-run` only checks metric cache availability and does not probe model caches. With training disabled, existing model caches are reused even if `SRALT_CACHE_REFRESH=1` is set; absent models remain missing. Run evaluations inside the project's Apptainer environment.
