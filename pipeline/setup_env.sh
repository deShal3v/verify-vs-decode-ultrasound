#!/bin/bash
# GPU-node environment, as on the paper's nodes (AWS g6 instances: NVIDIA L4 24 GB, Deep Learning AMI with an NVIDIA driver for
# CUDA 12.4): a Python 3.10 venv with the pinned packages of requirements-gpu.txt, NLTK data, the Hugging Face models the code
# loads, and Tang et al.'s repository at the commit the paper used.
# Usage: DATA_DIR=/data [VENV=~/venv] bash pipeline/setup_env.sh
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DATA_DIR=${DATA_DIR:?set DATA_DIR}; VENV=${VENV:-$HOME/venv}; TANG_REPO=${TANG_REPO:-$DATA_DIR/semantic-decoding}
export HF_HOME=${HF_HOME:-$DATA_DIR/hf}; mkdir -p "$DATA_DIR" "$HF_HOME"
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH=$HOME/.local/bin:$PATH
uv venv --python 3.10 "$VENV"; source "$VENV/bin/activate"
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124      # torch 2.4.1+cu124 comes from the PyTorch index
uv pip install -r "$ROOT/requirements-gpu.txt"
python -c "import nltk; [nltk.download(x, quiet=True) for x in ['wordnet','punkt','punkt_tab','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
[ -d "$TANG_REPO/.git" ] || git clone -q https://github.com/HuthLab/semantic-decoding.git "$TANG_REPO"
(cd "$TANG_REPO" && git checkout -q 6b25dd5720ced1c4199cd9b9149061683b77ba6b && echo "semantic-decoding at $(git log -1 --format=%H)")
python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer("sentence-transformers/gtr-t5-base")       # sentence embeddings (contrastive verifier, text side)
from bert_score import BERTScorer
BERTScorer(lang="en"); print("models cached")                   # BERTScore model (decoder evidence)
PY
