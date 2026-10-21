"""The decoder's goal in bits: log2 K_eff(tau_Tang), K_eff(tau) = 1 / P_null(window BERTScore >= tau), where tau_Tang is the
median 20-s window BERTScore of the decoder refitted on all 15.25 h of clean data (cell native_r15: decodes as well as the
published decoder) and P_null is pooled over the 56 test windows and the 200 LM-only nulls of that cell. Beyond the empirical
resolution (fewer than 10 null exceedances) the Gaussian tail is used (not needed for S1-S3). Same arithmetic as the paper's
Phase B analysis (final_b.keff). Also writes the 56 test-window anchors used by every readout.
Input: <models>/<S>/native_r15/bits20.npz (tang_bits.py <cell_dir> 200). Output: verifiers/analysis/reference/decoder_goal.json.
Usage: python decoder_goal.py [--models $DATA_DIR/models] [--out ../verifiers/analysis/reference/decoder_goal.json]"""
import os, sys, json, argparse, numpy as np
from scipy.stats import norm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vvd_paths as VP


def keff(bits20):
    z = np.load(bits20); null = z["bert_null"].astype(np.float64); real = z["bert_real"]
    mu, sd = null.mean(1), null.std(1)
    tau = float(np.median(real)); emp = float((null >= tau).mean()); gau = float(np.mean(norm.sf((tau - mu) / sd)))
    prim = float(np.log2(1 / emp)) if emp >= 10 / null.size else float(np.log2(1 / gau))
    return {"tau_tang": tau, "P_null_emp": emp, "P_null_gauss": gau, "log2K_primary": prim,
            "primary_source": "empirical" if emp >= 10 / null.size else "gaussian_tail", "n_null_samples": int(null.size),
            "anchors": [int(a) for a in z["anchors"]]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default=VP.MODELS)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "verifiers", "analysis", "reference", "decoder_goal.json"))
    a = ap.parse_args()
    out = {"definition": __doc__.split("Input:")[0].strip(), "per_subject": {S: keff(f"{a.models}/{S}/native_r15/bits20.npz") for S in ("S1", "S2", "S3")}}
    os.makedirs(os.path.dirname(a.out), exist_ok=True); json.dump(out, open(a.out, "w"), indent=1)
    print({S: round(v["log2K_primary"], 3) for S, v in out["per_subject"].items()})
