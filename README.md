# SRALT

SRALT is a social-relation level membership inference attack framework targeting long-tail users in social recommender systems. It evaluates privacy leakage under strict black-box settings and is built around three stages: a target stage for training and evaluating the victim recommender, a shadow stage for learning attack-oriented user representations, and an attack stage for predicting whether a social link exists between two users.

The codebase supports Ciao, Flickr, and Yelp datasets. The social target recommenders are DESIGN, GDMSR, and EGCL; LightGCN is available as a general (non-social) recommender baseline. The shadow models are SMIA, CSA, and DSA, and the final attack model is MLP. CSA, our proposed model, leverages residual vector quantization and semantic edge augmentation to learn high-quality representations for long-tail users. DSA, which removes the residual vector quantization module, serves as an ablation variant of CSA. SMIA serves as the baseline and employs two simple graph neural networks to learn user representations. The attack model is an MLP classifier that takes the representations of two users as input and predicts whether a social link exists between them.

MINER and En_MINER are also available as social-relation attack baselines, following the [SMIA authors' implementation](https://github.com/XuHao-bit/SMIA/tree/a7e79411244a574ba0b539510f105094a8956682). They use a dedicated shadow adapter and classify distance features between recommended item embeddings and another user's embedding.

## Project Structure

- `docs/`: research protocol, architecture, model design, and development guidance; start with [docs/README.md](docs/README.md)
- `scripts/`: scripts for end-to-end training and evaluation
- `sralt/`: package source code
- `sralt/configs/`: Hydra configuration files
- `data/`: dataset files
- `runs/`: Hydra outputs and experiment results
- `.cache/`: local cache for generated intermediate artifacts to speed up subsequent runs

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

To use the non-social LightGCN target, select `target/model=lightgcn`:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=lightgcn shadow/model=csa attack/model=mlp
```

LightGCN uses only user-item interactions. It follows the same data preprocessing, evaluation, and deployment-query protocol as the social targets. The existing comparison preset continues to select DESIGN, GDMSR, and EGCL.

All modifiable parameters are listed in the configuration files under `sralt/configs`. You can override any parameter by specifying its path and value on the command line. For example, to change the number of quantizers in the CSA shadow model, you can use the following command:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=csa attack/model=mlp shadow.model.attribute.num_quantizers=16
```

For a multirun sweep, add `-m` and override the values you want to explore. The following command sweeps over 3 datasets, 3 target models, 2 shadow models, and 5 random seeds, producing 90 sequential runs in total:

```bash
python ./scripts/pipeline.py -m target/data=ciao,flickr,yelp target/model=design,gdmsr,egcl shadow/model=smia,csa attack/model=mlp seed=42,43,44,45,46
```

Preset experiment configurations are stored in `sralt/configs/experiment`. They cover comparison, ablation, hyperparameter, and attacker-knowledge settings. For example, the following command runs the comparison experiment, which is equivalent to the previous multirun command:

```bash
python ./scripts/pipeline.py -m experiment=comparison
```

## Social MINER Baselines

Select the MINER shadow adapter together with the desired attack model:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=miner attack/model=miner
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=miner attack/model=enminer
```

MINER uses four distances (L1, L2, cosine, Bray-Curtis) from one user's K recommended items to the other user's embedding, giving `4K` attack features. En_MINER concatenates both directions, giving `8K` features. Both use the learned social embedding table and the SMIA authors' MLP architecture, unweighted binary cross entropy, and SGD with gradient clipping. Shadow training reuses SRALT's SMIA training settings; relation splits, negative sampling, validation selection, and All/Head/Tail metrics follow SRALT's protocol.

These reproduce the social baselines released with SMIA, which omit the original interaction-level MINER's knowledge graph, bilateral attack network, and cumulative learning. See [the reproduction notes](docs/miner-reproduction.md) for sources and implementation differences.

The dedicated sweep covers both attack modes, three datasets, three social target models, and five seeds (90 sequential runs):

```bash
python ./scripts/pipeline.py -m experiment=miner
```

## Output Structure

Hydra writes single-run outputs to `runs/outputs/YYYY-MM-DD/HH-MM-SS` and multirun sweeps to `runs/multirun/YYYY-MM-DD/HH-MM-SS`. The pipeline also saves target, shadow, and attack data/model checkpoints inside each run directory.
