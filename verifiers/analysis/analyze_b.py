"""Aggregate per-window verifier outputs into RESULTS.json (+ compact npz for figures).
Run on the instance after verify.py (all subjects) and product.py; optional Tang eval jsons.
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, math, glob, numpy as np
RES = f"{VP.WORK}/results"; TXT = os.environ.get("VFUS_TXT", f"{VP.WORK}/text")   # CPU prelim: TXT configurable
GRID = [("native", 15), ("native", 4), ("native", 1), ("m6", 15), ("m6", 4), ("m12", 15), ("m12", 4),
        ("m6c", 15), ("m6c", 4), ("m12c", 15), ("m12c", 4)]
SNRF = {"native": 1.0, "m6": 10 ** (-0.6), "m12": 10 ** (-1.2), "m6c": 10 ** (-0.6), "m12c": 10 ** (-1.2)}
NOISE = {"white": ["native", "m6", "m12"], "correlated": ["native", "m6c", "m12c"]}
DBINS = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2])
NB = 1000

def load_text():
    w6 = np.load(f"{TXT}/win6.npz"); s20 = np.load(f"{TXT}/seg20.npz")
    return {"6": {k: w6[k] for k in ["story", "anchor", "nw", "foil_d", "foil_valid", "foil_oov", "true_oov", "foil_names", "stories"]},
            "20": {k: s20[k] for k in ["story", "anchor", "nw", "foil_d", "foil_valid", "foil_oov", "true_oov", "foil_names", "stories"]}}

def cell_windows(z, rd, evs):
    """concatenate per-window arrays over eval blocks evs"""
    keys = ["s_true", "s_foil", "s_true_sh", "s_foil_sh", "s_true_tn", "s_foil_tn", "s_true_tn_sh", "s_foil_tn_sh", "logp1", "logp1_sh", "logp2", "logp2_sh", "rank2", "rank2_sh", "sel"]
    out = {k: [] for k in keys}; out["th"] = []; out["thtn"] = []; out["npool"] = []; out["block"] = []
    for bi, ev in enumerate(evs):
        if f"{rd}|{ev}|sel" not in z: continue
        n = len(z[f"{rd}|{ev}|sel"])
        for k in keys: out[k].append(z[f"{rd}|{ev}|{k}"])
        out["th"].append(np.full(n, float(z[f"{rd}|{ev}|th"]))); out["thtn"].append(np.full(n, float(z[f"{rd}|{ev}|thtn"]))); out["npool"].append(np.full(n, int(z[f"{rd}|{ev}|npool"]))); out["block"].append(np.full(n, bi))
    return {k: np.concatenate(v) for k, v in out.items()}

def metrics(w, tx, rng, boot=True):
    sel = w["sel"]; d = tx["foil_d"][sel]; v = tx["foil_valid"][sel]; names = list(tx["foil_names"])
    st, sf = w["s_true"][:, None].astype(np.float64), w["s_foil"].astype(np.float64)
    st_sh, sf_sh = w["s_true_sh"][:, None].astype(np.float64), w["s_foil_sh"].astype(np.float64)
    th = w["th"][:, None]
    win = (st > sf); hit = (w["s_true"] > w["th"]); fa = (sf > th)
    win_sh = (st_sh > sf_sh); hit_sh = (w["s_true_sh"] > w["th"])
    r = {"n_windows": int(len(sel)), "n_blocks": int(len(np.unique(w["block"])))}
    r["afc2_pooled"] = float(win[v].mean()); r["afc2_pooled_shuffled"] = float(win_sh[v].mean())
    r["hit_rate"] = float(hit.mean()); r["fa_pooled"] = float(fa[v].mean()); r["yesno_ba_pooled"] = (r["hit_rate"] + 1 - r["fa_pooled"]) / 2
    r["hit_rate_shuffled"] = float(hit_sh.mean())
    # primary yes/no: cohort T-normed score, threshold fixed on the calibration story
    hit_t = w["s_true_tn"] > w["thtn"]; fa_t = w["s_foil_tn"].astype(np.float64) > w["thtn"][:, None]
    r["hit_rate_tn"] = float(hit_t.mean()); r["fa_pooled_tn"] = float(fa_t[v].mean()); r["yesno_ba_tn_pooled"] = (r["hit_rate_tn"] + 1 - r["fa_pooled_tn"]) / 2
    r["hit_rate_tn_shuffled"] = float((w["s_true_tn_sh"] > w["thtn"]).mean())
    r["per_foil"] = {nm: {"afc2": float(win[:, j][v[:, j]].mean()) if v[:, j].any() else None,
                          "fa": float(fa[:, j][v[:, j]].mean()) if v[:, j].any() else None,
                          "yesno_ba": float((r["hit_rate"] + 1 - fa[:, j][v[:, j]].mean()) / 2) if v[:, j].any() else None,
                          "fa_tn": float(fa_t[:, j][v[:, j]].mean()) if v[:, j].any() else None,
                          "yesno_ba_tn": float((r["hit_rate_tn"] + 1 - fa_t[:, j][v[:, j]].mean()) / 2) if v[:, j].any() else None,
                          "dist_median": float(np.median(d[:, j][v[:, j]])) if v[:, j].any() else None,
                          "n": int(v[:, j].sum())} for j, nm in enumerate(names)}
    b = np.digitize(d, DBINS) - 1
    r["by_distance"] = []
    for k in range(len(DBINS) - 1):
        m = v & (b == k)
        if m.sum() < 20: continue
        r["by_distance"].append({"lo": float(DBINS[k]), "hi": float(DBINS[k + 1]), "n": int(m.sum()), "afc2": float(win[m].mean()),
                                 "fa": float(fa[m].mean()), "yesno_ba": float((r["hit_rate"] + 1 - fa[m].mean()) / 2),
                                 "fa_tn": float(fa_t[m].mean()), "yesno_ba_tn": float((r["hit_rate_tn"] + 1 - fa_t[m].mean()) / 2),
                                 "afc2_shuffled": float(win_sh[m].mean())})
    g1 = w["logp1"] - w["logp1_sh"]; g2 = w["logp2"] - w["logp2_sh"]
    rb = np.log2((w["npool"] + 1) / w["rank2"]); rb_sh = np.log2((w["npool"] + 1) / w["rank2_sh"])
    r["bits_p1_gain"] = float(g1.mean()); r["bits_p2_gain"] = float(g2.mean())
    r["rank_bits_p2"] = float(rb.mean()); r["rank_bits_p2_shuffled"] = float(rb_sh.mean()); r["rank_bits_p2_gain"] = float((rb - rb_sh).mean())
    r["top1_p2"] = float((w["rank2"] == 1).mean()); r["top10_p2"] = float((w["rank2"] <= 10).mean()); r["npool_median"] = float(np.median(w["npool"]))
    r["pool1_size"] = int(1 + v.sum(1).mean())
    if boot:
        blocks = w["block"]; ub = np.unique(blocks); bi = {u: np.where(blocks == u)[0] for u in ub}
        stats = {"afc2_pooled": [], "yesno_ba_pooled": [], "yesno_ba_tn_pooled": [], "bits_p1_gain": [], "bits_p2_gain": [], "rank_bits_p2_gain": []}
        for _ in range(NB):
            ii = np.concatenate([bi[u] for u in rng.choice(ub, len(ub))])
            vv = v[ii]; h = hit[ii].mean(); f = fa[ii][vv].mean()
            stats["afc2_pooled"].append(win[ii][vv].mean()); stats["yesno_ba_pooled"].append((h + 1 - f) / 2)
            stats["yesno_ba_tn_pooled"].append((hit_t[ii].mean() + 1 - fa_t[ii][vv].mean()) / 2)
            stats["bits_p1_gain"].append(g1[ii].mean()); stats["bits_p2_gain"].append(g2[ii].mean()); stats["rank_bits_p2_gain"].append((rb - rb_sh)[ii].mean())
        r["ci95"] = {k: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] for k, x in stats.items()}
    return r

def oov_table(tx):
    names = list(tx["foil_names"]); v = tx["foil_valid"]; o = tx["foil_oov"]
    return {"true": float(np.mean(tx["true_oov"])), **{nm: float(np.nanmean(np.where(v[:, j], o[:, j], np.nan))) for j, nm in enumerate(names)}}

def main():
    rng = np.random.default_rng(0); tx = load_text(); stories = list(tx["6"]["stories"])
    TEST = "wheretheressmoke"; out = {"cells": {}, "oov_rate": {rd: oov_table(tx[rd]) for rd in ("6", "20")}}
    train_st = stories[:82]
    for subj in ["S1", "S2", "S3"]:
        cj = f"{RES}/{subj}_cells.json"
        if not os.path.exists(cj): continue
        cells = json.load(open(cj))
        for cond, k in GRID:
            cell = f"{cond}_r{k}"
            if cell not in cells or not os.path.exists(f"{RES}/{subj}_{cell}.npz"): continue
            z = np.load(f"{RES}/{subj}_{cell}.npz"); meta = cells[cell]
            entry = {kk: meta[kk] for kk in ["cond", "dB", "rung_sessions", "hours", "n_train_stories", "q", "lam", "n_params_trainable", "n_params_pca", "n_models", "gpu_s", "wall_s"]}
            entry["n_hparam_trials"] = len(meta["trials"]); entry["snr_factor"] = SNRF[cond]; entry["hours_x_snr"] = meta["hours"] * SNRF[cond]
            for rd in ("6", "20"):
                entry[f"loso_{rd}"] = metrics(cell_windows(z, rd, train_st), tx[rd], rng)
                entry[f"test_rep0_{rd}"] = metrics(cell_windows(z, rd, ["test_rep0"]), tx[rd], rng, boot=False)
                reps = [metrics(cell_windows(z, rd, [f"test_rep{r}"]), tx[rd], rng, boot=False) for r in range(10)]
                entry[f"test_10reps_{rd}"] = {m: [float(np.mean([x[m] for x in reps])), float(np.std([x[m] for x in reps]))]
                                             for m in ["afc2_pooled", "yesno_ba_pooled", "yesno_ba_tn_pooled", "hit_rate_tn", "fa_pooled_tn", "bits_p1_gain", "bits_p2_gain", "rank_bits_p2_gain", "top1_p2", "hit_rate", "fa_pooled"]}
            out["cells"][f"{subj}_{cell}"] = entry
            print(subj, cell, {rd: (round(entry[f"loso_{rd}"]["yesno_ba_tn_pooled"], 3), round(entry[f"loso_{rd}"]["yesno_ba_pooled"], 3), round(entry[f"loso_{rd}"]["afc2_pooled"], 3), round(entry[f"loso_{rd}"]["bits_p1_gain"], 3), round(entry[f"loso_{rd}"]["rank_bits_p2_gain"], 3)) for rd in ("6", "20")}, flush=True)
    # learning-curve slopes (native, 3 rungs) and collapse test
    lc = {}
    for rd in ("6", "20"):
        for met in ["bits_p1_gain", "bits_p2_gain", "rank_bits_p2_gain", "yesno_ba_tn_pooled", "yesno_ba_pooled", "afc2_pooled"]:
            sl = {}
            for subj in ["S1", "S2", "S3"]:
                pts = [(out["cells"][f"{subj}_native_r{k}"]["hours"], out["cells"][f"{subj}_native_r{k}"][f"loso_{rd}"][met]) for k in (1, 4, 15) if f"{subj}_native_r{k}" in out["cells"]]
                if len(pts) >= 2:
                    x = np.log10([p[0] for p in pts]); y = np.array([p[1] for p in pts]); sl[subj] = float(np.polyfit(x, y, 1)[0])
                    if len(pts) == 3: sl[subj + "_curvature"] = float(np.polyfit(x, y, 2)[0])
            lc[f"{rd}s_{met}"] = sl
    out["learning_curve_slopes_per_decade"] = lc
    col = {}
    for nm, conds in NOISE.items():
        sfx = "c" if nm == "correlated" else ""
        for rd in ("6", "20"):
            for met in ["bits_p1_gain", "rank_bits_p2_gain", "yesno_ba_tn_pooled", "afc2_pooled"]:
                per = {}
                for subj in ["S1", "S2", "S3"]:
                    P = [(c["hours_x_snr"], c["hours"], c["snr_factor"], c[f"loso_{rd}"][met]) for kk, c in out["cells"].items()
                         if kk.startswith(subj + "_") and c["cond"] in conds]
                    if len(P) < 5: continue
                    P = np.array(P); x = np.log10(P[:, 0]); y = P[:, 3]
                    c1 = np.polyfit(x, y, 1); res1 = y - np.polyval(c1, x)
                    Xm = np.c_[np.ones(len(P)), np.log10(P[:, 1]), np.log10(P[:, 2])]; beta, *_ = np.linalg.lstsq(Xm, y, rcond=None); res2 = y - Xm @ beta
                    def g(cell): return out["cells"].get(f"{subj}_{cell}", {}).get(f"loso_{rd}", {}).get(met)
                    pairs = {"15h_-6dB_vs_4h_0dB": [g(f"m6{sfx}_r15"), g("native_r4")],
                             "15h_-12dB_vs_4h_-6dB_vs_1h_0dB": [g(f"m12{sfx}_r15"), g(f"m6{sfx}_r4"), g("native_r1")]}
                    per[subj] = {"n_points": int(len(P)), "single_curve_slope": float(c1[0]), "rms_single": float(np.sqrt(np.mean(res1 ** 2))),
                                 "slope_hours": float(beta[1]), "slope_snr": float(beta[2]), "rms_two_slopes": float(np.sqrt(np.mean(res2 ** 2))),
                                 "beta_snr_over_hours": float(beta[2] / beta[1]) if beta[1] != 0 else None, "matched_products": pairs}
                col[f"{nm}|{rd}s_{met}"] = per
    # white vs correlated at equal nominal dB: loss relative to native at the same rung
    cmp_ = {}
    for rd in ("6", "20"):
        for met in ["bits_p1_gain", "rank_bits_p2_gain", "yesno_ba_tn_pooled"]:
            for subj in ["S1", "S2", "S3"]:
                for k in (15, 4):
                    nat = out["cells"].get(f"{subj}_native_r{k}", {}).get(f"loso_{rd}", {}).get(met)
                    for db in ("6", "12"):
                        w = out["cells"].get(f"{subj}_m{db}_r{k}", {}).get(f"loso_{rd}", {}).get(met)
                        c = out["cells"].get(f"{subj}_m{db}c_r{k}", {}).get(f"loso_{rd}", {}).get(met)
                        if None in (nat, w, c): continue
                        cmp_[f"{rd}s|{met}|{subj}|r{k}|-{db}dB"] = {"native": nat, "white": w, "correlated": c, "loss_white": nat - w, "loss_correlated": nat - c,
                                                                     "loss_ratio_corr_over_white": (nat - c) / (nat - w) if nat != w else None}
    out["noise_model_comparison"] = cmp_
    nf = {}
    for subj in ["S1", "S2", "S3"]:
        p = f"{VP.WORK}/{subj}/noise_fit.json"
        if os.path.exists(p): nf[subj] = {k: v for k, v in json.load(open(p)).items() if k not in ("bins_mm", "corr")}
    out["noise_fit"] = nf
    out["collapse"] = col
    pt = f"{RES}/product_tests.json"
    if os.path.exists(pt):
        P = json.load(open(pt))
        out["product_tests"] = {s: {"imagined_timing_free": {k: v for k, v in P[s]["imagined"]["timing_free"].items() if k != "scores"},
                                    "imagined_timing_rule": {k: v for k, v in P[s]["imagined"]["timing_rule"].items() if k != "scores"},
                                    "multispeaker": {r: {k: v for k, v in d.items() if k != "scores20"} for r, d in P[s]["multispeaker"].items()}} for s in P}
    pc = f"{RES}/product_checks.json"
    if os.path.exists(pc): out["product_checks"] = json.load(open(pc))
    tang = {}
    for f in sorted(glob.glob(f"{VP.DATA}/tang_scores/S*/perceived_speech/wheretheressmoke_eval.json")):
        d = json.load(open(f)); tang[d["subject"]] = d
    out["tang_reproduction"] = tang
    tr = f"{RES}/tang_distance.json"
    if os.path.exists(tr): out["tang_on_distance_axis"] = json.load(open(tr))
    json.dump(out, open(f"{RES}/RESULTS.json", "w"), indent=1)
    print("wrote RESULTS.json")

if __name__ == "__main__":
    main()
