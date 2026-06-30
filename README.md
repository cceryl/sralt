# SRALT

SRALT is a social relation-level membership inference attack framework for long-tail users in social recommender systems. It evaluates privacy leakage under strict black-box settings and is built around three stages: a target stage that trains and evaluates the victim recommender, a shadow stage that learns attack-oriented user representations, and an attack stage that predicts whether a social link exists between two users.

The codebase supports Ciao, Flickr, and Yelp. The target recommenders are DESIGN, GDMSR, and EGCL. The shadow models are SMIA, CSA, and DSA, and the final attack model is MLP. CSA is the proposed model that leverages residual vector quantization and semantic edge augmentation to learn high-quality representations for long-tail users. DSA is a variant of CSA that removes the residual vector quantization module, and SMIA is a baseline that uses a simple graph neural network to learn user representations.

## Project Structure

- `scripts/`: end-to-end training and evaluation entry point
- `sralt/`: package source code
- `sralt/configs/`: Hydra configuration files
- `data/`: dataset files
- `runs/`: Hydra outputs and experiment results
- `.cache/`: local cache for generated intermediate artifacts to speed up subsequent runs

## Installation

The repository targets Linux and the provided environment uses Python 3.12.13. Make sure `uv` is installed first, then run:

```bash
bash install.sh
```

The script creates a Python 3.12.13 virtual environment, installs the required packages, and installs this project in editable mode.

You may need to activate the virtual environment before running the pipeline:

```bash
source .venv/bin/activate
```

## Quick Start

The pipeline is configured with Hydra. For a single run, you must at least specify `target/data`, `target/model`, `shadow/model`, and `attack/model` explicitly. A runnable example is:

```bash
python ./scripts/pipeline.py target/data=ciao target/model=design shadow/model=csa attack/model=mlp
```

All modifiable parameters are listed in the configuration files under `sralt/configs`. You can override any parameter by specifying its path and value in the command line. For example, to change the number of quantizers in the CSA shadow model, you can run:

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

## Output Structure

Hydra writes single-run outputs to `runs/outputs/YYYY-MM-DD/HH-MM-SS` and multirun sweeps to `runs/multirun/YYYY-MM-DD/HH-MM-SS`. The pipeline also saves target, shadow, and attack data/model checkpoints inside each run directory.
