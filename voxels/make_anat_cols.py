"""The readouts' voxel set: one anatomical preset per participant, chosen from anatomy alone before any story data were used.
V_anat(S) = auditory  U  language (LanA probability >= 0.20)  U  speech-motor (HCP-MMP1 6v, 6r, 55b, 44, 45), both hemispheres,
the a-priori area sets of atlas_sets.py (auditory = FreeSurfer aparc transversetemporal + superiortemporal U Tang et al.'s
released auditory ROI; speech-motor U Tang et al.'s released speech ROI). Columns are full-grid hf5 columns (ds003020 order).
Sizes in the paper: S1 9,694, S2 12,113, S3 11,682 voxels. The decoder, encode-then-compare and the contrastive verifier all
use exactly these voxels (intersected with an ultrasound condition's reachable set).
Input: $WORK_DIR/<S>/areas.json (atlas_sets.py). Output: $WORK_DIR/inputs/<S>/anat_cols.npy (int64, sorted).
Usage: python make_anat_cols.py [S1 S2 S3]"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vvd_paths as VP

SETS = ("auditory", "language_LanA", "speech_motor")
PAPER_SIZES = {"S1": 9694, "S2": 12113, "S3": 11682}

if __name__ == "__main__":
    for S in sys.argv[1:] or ("S1", "S2", "S3"):
        A = json.load(open(f"{VP.WORK}/{S}/areas.json"))["sets"]
        cols = np.union1d(np.union1d(np.asarray(A[SETS[0]], np.int64), np.asarray(A[SETS[1]], np.int64)), np.asarray(A[SETS[2]], np.int64))
        os.makedirs(f"{VP.INPUTS}/{S}", exist_ok=True); np.save(f"{VP.INPUTS}/{S}/anat_cols.npy", cols.astype(np.int64))
        print(S, len(cols), "voxels", "(paper: %d)" % PAPER_SIZES[S], {k: len(A[k]) for k in SETS})
