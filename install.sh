uv venv --python python3.12.13
source .venv/bin/activate

uv pip install torch torchvision torchaudio torch_geometric torch_scatter torch_sparse \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --find-links https://data.pyg.org/whl/torch-2.12.0+cu126.html

uv pip install torchmetrics tensorboard nvidia-ml-py cuda-python vector_quantize_pytorch

uv pip install matplotlib seaborn pandas scikit-learn

uv pip install debugpy tqdm colorlog hydra-core openpyxl

uv pip install -e .
