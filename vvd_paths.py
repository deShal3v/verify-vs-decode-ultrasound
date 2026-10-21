"""Paths used by every script in this repository (one place).

Set two environment variables (or accept the defaults, which are relative to the repository):
  DATA_DIR  raw downloads: OpenNeuro ds003020 (and the ds004510 test TextGrid), Tang et al.'s Box data, atlases.
            Default: <repo>/data.  On the GPU nodes of the paper this was /data.
  WORK_DIR  everything derived: per-subject response matrices, noise maps, node inputs, model fits, scores, summaries.
            Default: <repo>/work.  On the GPU nodes of the paper this was /data/work.
Optional:
  TANG_REPO clone of https://github.com/HuthLab/semantic-decoding at commit 6b25dd5720ced1c4199cd9b9149061683b77ba6b
            (default: $DATA_DIR/semantic-decoding).
  HF_HOME   Hugging Face cache (default: $DATA_DIR/hf)."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.expanduser(os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))))
WORK = os.path.abspath(os.path.expanduser(os.environ.get("WORK_DIR", os.path.join(ROOT, "work"))))
TANG_REPO = os.path.abspath(os.path.expanduser(os.environ.get("TANG_REPO", os.path.join(DATA, "semantic-decoding"))))
TANG_CODE = os.path.join(TANG_REPO, "decoding")
TANG_COMMIT = "6b25dd5720ced1c4199cd9b9149061683b77ba6b"
HF_HOME = os.path.abspath(os.path.expanduser(os.environ.get("HF_HOME", os.path.join(DATA, "hf"))))

DS = os.path.join(DATA, "ds003020")                 # OpenNeuro ds003020 (derivatives/ contents: preprocessed_data, TextGrids, ...)
TANG_BOX = os.path.join(DATA, "tang_box", "x")      # Tang et al. Box data unzipped: data_lm/, data_train/, data_test/
ATLAS = os.path.join(DATA, "atlas")                 # lh/rh.HCP-MMP1.annot, lana/FS/{LH,RH}_LanA_n804.nii.gz, nilearn/
PYCORTEX = os.path.join(DATA, "pycortex")           # <UTS>/mask_thick.nii.gz (copied from ds003020 pycortex-db)
MODELS = os.path.join(DATA, "models")               # per-cell model fits (GPU pipeline)
INPUTS = os.path.join(WORK, "inputs")               # CPU-built per-subject inputs for the GPU pipeline (us_maps.npz, anat_cols.npy, ...)
LOGS = os.path.join(WORK, "logs")
