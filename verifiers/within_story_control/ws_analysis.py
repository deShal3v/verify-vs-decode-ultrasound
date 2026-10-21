"""Within-story control (2026-10-21, Omer's ISBI edit list item 3): contrastive verifier, clean and -12 dB, 15.25 h, S1-S3 (verify_ws.py on the
nodes, collected with pipeline/collect_out_e.sh). Per held-out story and window, rank bits log2((P+1)/rank) - log2((P+1)/rank_shifted) as summ_e4.per_story,
(1) with the paper's other-story pool (reproduction check against the paper's value from GRID_E4_s7.json) and (2) with the same-story pool;
mean over all windows; story-cluster bootstrap with the grid's draws (default_rng(2000 + subject index).integers(0, 82, (1000, 82)), the first
draw of grid_e4.py). Output: <out>/within_story_control.csv (one row per participant and condition, plus the mean rows).
--rescue = the directory with one sub-directory per GPU machine (pipeline/collect_out_e.sh), as for verifiers/analysis/summarize.py.
Run: python ws_analysis.py --rescue $WORK_DIR/rescue/ws --grid results/GRID_E4_s7.json --out results"""
import os, glob, json, csv, argparse, numpy as np
_ap = argparse.ArgumentParser(); _ap.add_argument("--rescue", required=True); _ap.add_argument("--grid", required=True); _ap.add_argument("--out", required=True)
_A = _ap.parse_args(); RS = os.path.abspath(_A.rescue); OUT = os.path.abspath(_A.out); os.makedirs(OUT, exist_ok=True)
G = json.load(open(_A.grid))["per_subject"]
COND = {"native_n82": ("clean", "native"), "m12c_n82": ("-12 dB", "m12c")}
rows = []
for si, S in enumerate(("S1", "S2", "S3")):
    STORY_II = np.random.default_rng(2000 + si).integers(0, 82, (1000, 82))
    for cell, (label, gcond) in COND.items():
        z = np.load(glob.glob(f"{RS}/*/{S}/work/results_C_us/{S}_ws_{cell}.npz")[0])
        info = list(json.load(open(glob.glob(f"{RS}/*/{S}/work/results_C_us/{S}_ws_{cell}_cell.json")[0])).values())[0]
        st = info["stories"]; assert len(st) == 82
        rb = lambda r, rs, P: np.log2((P + 1) / r) - np.log2((P + 1) / rs)
        out = {}
        for tag, (r, rs, P) in (("other", ("rank2", "rank2_sh", "npool")), ("same", ("rank2_ws", "rank2_ws_sh", "npool_ws"))):
            per = [rb(z[f"{s}|{r}"].astype(float), z[f"{s}|{rs}"].astype(float), z[f"{s}|{P}"].astype(float)) for s in st]
            n = np.array([len(p) for p in per], float); sm = np.array([p.sum() for p in per])
            bb = sm[STORY_II].sum(1) / n[STORY_II].sum(1)
            out[tag] = (float(sm.sum() / n.sum()), float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5)), int(n.sum()))
        Pws = np.concatenate([z[f"{s}|npool_ws"] for s in st]); Po = int(z[f"{st[0]}|npool"])
        paper = next(p[2] for p in G[S]["conditions"][gcond]["curves"]["ver_C"] if p[0] == "n82")
        rows.append({"participant": S, "condition": label, "training_h": 15.25, "n_stories": 82, "n_windows": out["same"][3],
                     "bits_same_story": round(out["same"][0], 3), "same_ci95_lo": round(out["same"][1], 3), "same_ci95_hi": round(out["same"][2], 3),
                     "bits_other_story_rerun": round(out["other"][0], 3), "other_ci95_lo": round(out["other"][1], 3), "other_ci95_hi": round(out["other"][2], 3),
                     "bits_other_story_paper": round(float(paper), 3), "pool_same_story_median": int(np.median(Pws)), "pool_other_story": Po,
                     "q": info["q"], "lambda": info["lam"]})
for label in ("clean", "-12 dB"):
    rr = [r for r in rows if r["condition"] == label]
    rows.append({"participant": "mean", "condition": label, "training_h": 15.25, "n_stories": 82, "n_windows": sum(r["n_windows"] for r in rr),
                 **{k: round(float(np.mean([r[k] for r in rr])), 3) for k in ("bits_same_story", "bits_other_story_rerun", "bits_other_story_paper")}})
keys = list(rows[0].keys())
with open(f"{OUT}/within_story_control.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k, "") for k in keys}) for r in rows]
for r in rows: print({k: r.get(k) for k in ("participant", "condition", "bits_same_story", "same_ci95_lo", "same_ci95_hi", "bits_other_story_rerun", "bits_other_story_paper", "pool_same_story_median")})
