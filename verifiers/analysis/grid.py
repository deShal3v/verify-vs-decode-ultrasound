"""Phase E4 grid (2026-10-20): grid_e3.py on Phase E + E2 + E3 + E4 merged (adds results_e4: the K = 2 ultrasound conditions usL90k2, usB90k2,
usL50k2, usL50k2u6/u12/u18 and the decoder at x63, m18c, which also enters the decoder's noise-survived ladder). Original Phase E3 docstring follows.
Phase E3 grid: grid_e2.py on Phase E + E2 + E3 merged (results_e + results_e2 + results_e3), with the E3 coded-excitation conditions added.
Phase E2 grid: grid_e.py on Phase E + Phase E2 merged (results_e + results_e2; Phase E2 adds n53/n59/n70/n76). SLOPE_SET env:
"7" = the registered Phase E slope rungs (must reproduce GRID_E.json), "11" = n4 ... n82 with the new rungs (secondary). Original docstring:
Phase E grid (DESIGN_E.md 4-5): per subject, condition and readout, the learning curve on the story-doubling ladder, crossings
(hours to the level; noise survived at fixed training), slopes and the registered slope test.
Readout scoring and bootstrap draws are Phase C final_c.py's, re-implemented here verbatim for the Phase E cells only:
  C = held-out LOSO 20 s rank bits (story-cluster bootstrap); E-t-C = test story, mean of 10 single noisy repeats (moving-block
  bootstrap, blocks of 3 segments); decoder = 56 test segments, strict count on 4,000 nulls (primary), Gaussian-tail on 4,000
  (friendly), counted on 200 (moving-block bootstrap, same draws as E-t-C -> paired). Seeds: default_rng(2000 + subject index).
Slope = least squares of bits on log10(hours) over n4 ... n82 (7 rungs). d = s_ETC - s_dec (paired); d_C = s_C - s_dec.
Release note: the results directories are given on the command line in the paper's merge order (Phase E first; later phases
add cells); the decoder goal (log2 K_eff of the published decoder's median window, Phase B) and the 56 test windows come from
reference/decoder_goal.json (decoder/decoder_goal.py), the equivalent-noise column from $WORK_DIR/geometry/us_conditions_summary.json.
Run: SLOPE_SET=7 python grid.py --results $WORK_DIR/results/E $WORK_DIR/results/E2 $WORK_DIR/results/E3 $WORK_DIR/results/E4 --out $WORK_DIR/analysis
     -> GRID_E4_s7.json (the paper's grid)"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, argparse, numpy as np
_ap = argparse.ArgumentParser(); _ap.add_argument("--results", nargs="+", required=True); _ap.add_argument("--out", required=True)
_ap.add_argument("--reference", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "decoder_goal.json"))
_ap.add_argument("--geo", default=f"{VP.WORK}/geometry/us_conditions_summary.json"); _A = _ap.parse_args()
RFS = [os.path.abspath(x) for x in _A.results]; AO = os.path.abspath(_A.out); os.makedirs(AO, exist_ok=True)
SUBJ = ["S1", "S2", "S3"]; NB = 1000; BLOCK = 3; INV_LN2 = 1 / np.log(2)
RUNGS = ["n1", "n2", "n4", "n8", "n16", "n32", "n48", "n56", "n64", "n73", "n82"]; SLOPE_SET = os.environ.get("SLOPE_SET", "7")
SLOPE_RUNGS = ["n4", "n8", "n16", "n32", "n48", "n64", "n82"] if SLOPE_SET == "7" else RUNGS[2:]
CONDS = [("native", 0), ("m6c", -6), ("m12c", -12), ("m18c", -18), ("usL90", None), ("usB90", None), ("usL50", None),
         ("usL50u6", None), ("usL50u12", None), ("usL50u18", None), ("usL50c13", None), ("usL50i1365", None), ("usL50c65", None),
         ("usL50c13cl", None), ("usL50i1365cl", None), ("usL50c65cl", None),
         ("usL90k2", None), ("usB90k2", None), ("usL50k2", None), ("usL50k2u6", None), ("usL50k2u12", None), ("usL50k2u18", None)]
REF = json.load(open(_A.reference))["per_subject"]; GEO = json.load(open(_A.geo)) if os.path.exists(_A.geo) else {}


def ci(a, q=(2.5, 97.5)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return [float(np.percentile(a, q[0])), float(np.percentile(a, q[1]))] if len(a) else [None, None]


def cross_up(xs, ys, thr):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    if y[0] >= thr: return {"x": None, "interval": [-np.inf, float(x[0])], "type": "at/above at the shortest rung"}
    for j in range(1, len(x)):
        if y[j] >= thr:
            f = (thr - y[j - 1]) / (y[j] - y[j - 1]); v = x[j - 1] + f * (x[j] - x[j - 1]); return {"x": float(v), "interval": [float(v), float(v)], "type": "measured"}
    return {"x": None, "interval": [float(x[-1]), np.inf], "type": "below at the longest rung"}


def cross_down(xs, ys, thr):
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    if y[0] < thr: return {"x": None, "interval": [float(x[0]), np.inf], "type": f"below already at {x[0]:g}"}
    for j in range(1, len(x)):
        if y[j] < thr:
            f = (y[j - 1] - thr) / (y[j - 1] - y[j]); v = x[j - 1] + f * (x[j] - x[j - 1]); return {"x": float(v), "interval": [float(v), float(v)], "type": "measured"}
    return {"x": None, "interval": [-np.inf, float(x[-1])], "type": f"still above at {x[-1]:g}"}


def boot(fn, xs, Yb, thr):
    return np.array([(lambda c: c["x"] if c["x"] is not None else np.nan)(fn(xs, Yb[:, j], thr)) for j in range(Yb.shape[1])], float)


def gap(a, b, ab=None, bb=None):
    """a - b with interval arithmetic; CI from paired draws when both are measured"""
    lo = a["interval"][0] - b["interval"][1]; hi = a["interval"][1] - b["interval"][0]
    e = {"gap": float(hi) if lo == hi else None, "interval": [lo, hi]}
    if a["type"] == "measured" and b["type"] == "measured" and ab is not None and bb is not None: e["ci95"] = ci(ab - bb)
    return e


def slope(lx, y): return float(np.polyfit(lx, y, 1)[0])


def clean2(o):
    if isinstance(o, dict): return {str(k): clean2(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)): return [clean2(v) for v in o]
    if isinstance(o, (np.floating, float)): return ("+inf" if o > 0 else "-inf") if np.isinf(o) else (None if np.isnan(o) else float(o))
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.ndarray): return clean2(o.tolist())
    return o


OUT = {"design": "phaseE/DESIGN_E.md", "per_subject": {}}
for si, S in enumerate(SUBJ):
    RF = RFS[0]
    if not os.path.exists(f"{RF}/{S}_summary.json"): continue
    C_ = json.load(open(f"{RF}/{S}_summary.json")); CW = dict(np.load(f"{RF}/{S}_windows.npz"))
    for RFk in RFS[1:]:                                                       # later phases add their cells (same rule as the paper's grid_e2/e3/e4)
        if os.path.exists(f"{RFk}/{S}_summary.json"):
            Ck = json.load(open(f"{RFk}/{S}_summary.json")); CW.update(dict(np.load(f"{RFk}/{S}_windows.npz")))
            for fam in ("C", "ETC", "Tang"): C_.setdefault(fam, {}).update(Ck.get(fam, {}))
    BAR = REF[S]["log2K_primary"]
    rng = np.random.default_rng(2000 + si); STORY_II = rng.integers(0, 82, (NB, 82))
    ANCH = np.array(REF[S]["anchors"]); nW = len(ANCH); nb = int(np.ceil(nW / BLOCK))
    BLK = (rng.integers(0, nW - BLOCK + 1, (NB, nb))[:, :, None] + np.arange(BLOCK)).reshape(NB, -1)[:, :nW]

    def ver(fam, cell):
        if fam == "C":
            e = C_.get("C", {}).get(cell)
            if not e or "loso_20" not in e: return None
            st = {k: np.asarray(v, float) for k, v in e["loso_20_story"].items()}
            bb = st["rb_sum"][STORY_II].sum(1) / st["n"][STORY_II].sum(1)
            return {"bits": e["loso_20"]["rank_bits_p2_gain"], "_bb": bb, "hours": e.get("hours"), "n_vox": e.get("V_set")}
        e = C_.get("ETC", {}).get(cell)
        if not e or f"ETC|{cell}|20|test10|rb" not in CW: return None
        an = CW[f"ETC|{cell}|20|test_rep0|anchor"]; pos = {int(x): j for j, x in enumerate(an)}; ix = [pos[int(x)] for x in ANCH]
        rbm = CW[f"ETC|{cell}|20|test10|rb"][:, ix].astype(float).mean(0)
        return {"bits": float(rbm.mean()), "_bb": rbm[BLK].mean(1), "hours": e.get("hours"), "n_vox": e.get("n_voxels")}

    def dec(cell, kind):
        e = C_.get("Tang", {}).get(cell)
        if not e or f"T|{cell}|bits" not in CW: return None
        h = (e.get("fit_info") or {}).get("hours")
        if kind in ("empBig", "gaussBig"):
            if f"T|{cell}|bitsBig" not in CW: return None
            an = CW[f"T|{cell}|anchors_big"]; pos = {int(a): j for j, a in enumerate(an)}; ix = [pos[int(a)] for a in ANCH]
            arr = (CW[f"T|{cell}|bitsBig"] - CW[f"T|{cell}|bits_null_mean_big"])[ix] if kind == "empBig" else (CW[f"T|{cell}|bitsGBig"] - INV_LN2)[ix]
        else:
            an = CW[f"T|{cell}|anchors"]; pos = {int(a): j for j, a in enumerate(an)}; ix = [pos[int(a)] for a in ANCH]
            arr = (CW[f"T|{cell}|bits"] - CW[f"T|{cell}|bits_null_mean"])[ix] if kind == "emp200" else (CW[f"T|{cell}|bitsG"] - INV_LN2)[ix]
        return {"bits": float(arr.mean()), "_bb": arr[BLK].mean(1), "hours": h, "n_vox": (e.get("fit_info") or {}).get("n_selected")}

    O = {"bar": BAR, "conditions": {}}
    for cond, db in CONDS:
        e = {"dB": db, "equivalent_plain_noise_dB": (GEO.get(S, {}).get("conds", {}).get(cond, {}) or {}).get("equivalent_plain_noise_dB") if cond.startswith("us") else db, "curves": {}, "cross": {}, "slope": {}}
        R = {}
        for name, fn in (("ver_C", lambda c: ver("C", c)), ("ver_ETC", lambda c: ver("ETC", c)), ("dec_empBig", lambda c: dec(c, "empBig")),
                         ("dec_gaussBig", lambda c: dec(c, "gaussBig")), ("dec_emp200", lambda c: dec(c, "emp200"))):
            pts = [(r, fn(f"{cond}_{r}")) for r in RUNGS]; pts = [(r, p) for r, p in pts if p and p.get("hours")]
            R[name] = pts; e["curves"][name] = [(r, p["hours"], p["bits"], p.get("n_vox")) for r, p in pts]
            if not pts: continue
            lx = [np.log10(p["hours"]) for _, p in pts]; ys = [p["bits"] for _, p in pts]; Yb = np.stack([p["_bb"] for _, p in pts], 0)
            thr = 1.0 if name.startswith("ver") else BAR
            c = cross_up(lx, ys, thr) if len(pts) >= 2 else {"x": None, "interval": [-np.inf, lx[0]] if ys[0] >= thr else [lx[0], np.inf], "type": "single"}
            cb = boot(cross_up, lx, Yb, thr) if c["type"] == "measured" else None
            e["cross"][name] = {**c, "hours_interval": [10 ** v if np.isfinite(v) else (0.0 if v < 0 else np.inf) for v in c["interval"]], "_b": cb}
            sp = [(r, p) for r, p in pts if r in SLOPE_RUNGS]
            if len(sp) == len(SLOPE_RUNGS):
                lxs = [np.log10(p["hours"]) for _, p in sp]; Ys = np.stack([p["_bb"] for _, p in sp], 0)
                e["slope"][name] = {"s": slope(lxs, [p["bits"] for _, p in sp]), "ci95": ci([slope(lxs, Ys[:, j]) for j in range(NB)]),
                                    "_b": np.array([slope(lxs, Ys[:, j]) for j in range(NB)]),
                                    "early_n4_n16": slope(lxs[:3], [p["bits"] for _, p in sp][:3]), "late_n16_n82": slope(lxs[2:], [p["bits"] for _, p in sp][2:])}
        # the registered slope test and the hours ratio
        T = {}
        for v in ("ver_ETC", "ver_C"):
            for k in ("dec_empBig", "dec_gaussBig"):
                if v in e["slope"] and k in e["slope"]:
                    dd = e["slope"][v]["_b"] - e["slope"][k]["_b"]; c95 = ci(dd); dv = e["slope"][v]["s"] - e["slope"][k]["s"]
                    T[f"{v}-{k}"] = {"d": dv, "ci95": c95, "paired": v == "ver_ETC",
                                     "verdict": ("verifier steeper" if c95[0] > 0 else "decoder steeper" if c95[1] < 0 else "equal (|d|<=0.5, CI within +-1)" if abs(dv) <= 0.5 and -1 <= c95[0] and c95[1] <= 1 else "inconclusive")}
                if v in e["cross"] and k in e["cross"]:
                    g = gap(e["cross"][k], e["cross"][v], e["cross"][k].get("_b"), e["cross"][v].get("_b"))
                    T[f"hours_ratio {k}/{v}"] = {"ratio": 10 ** g["gap"] if g["gap"] is not None else None, "interval": [10 ** x if np.isfinite(x) else (0.0 if x < 0 else np.inf) for x in g["interval"]],
                                                 "ci95": [10 ** x for x in g["ci95"]] if g.get("ci95") and g["ci95"][0] is not None else None}
        e["tests"] = T
        O["conditions"][cond] = e
    # noise survived at fixed training (plain ladder only), per rung
    NS = {}
    for r in RUNGS:
        row = {}
        for name, fn, thr in (("ver_C", lambda c: ver("C", c), 1.0), ("ver_ETC", lambda c: ver("ETC", c), 1.0), ("dec_empBig", lambda c: dec(c, "empBig"), BAR), ("dec_gaussBig", lambda c: dec(c, "gaussBig"), BAR)):
            pts = [(db, fn(f"{cond}_{r}")) for cond, db in CONDS if db is not None]; pts = [(db, p) for db, p in pts if p]
            if len(pts) >= 2: row[name] = {**cross_down([p[0] for p in pts], [p[1]["bits"] for p in pts], thr), "points": [(db, p["bits"]) for db, p in pts]}
            elif len(pts) == 1: row[name] = {"x": None, "interval": [-np.inf, pts[0][0]] if pts[0][1]["bits"] >= thr else [pts[0][0], np.inf], "type": "single", "points": [(pts[0][0], pts[0][1]["bits"])]}
        NS[r] = row
    O["noise_survived"] = NS
    OUT["per_subject"][S] = O
json.dump(clean2(OUT), open(f"{AO}/GRID_E4_s{SLOPE_SET}.json", "w"), indent=1)
for S, O in OUT["per_subject"].items():
    print(f"== {S} bar {O['bar']:.2f}")
    for cond, e in O["conditions"].items():
        hh = {k: [None if not np.isfinite(x) else round(x, 2) for x in v["hours_interval"]] for k, v in e["cross"].items() if k in ("ver_C", "ver_ETC", "dec_empBig")}
        sl = {k: round(v["s"], 2) for k, v in e["slope"].items() if k in ("ver_C", "ver_ETC", "dec_empBig")}
        tt = {k: (round(v["d"], 2), [round(x, 2) for x in v["ci95"]], v["verdict"]) for k, v in e["tests"].items() if k.endswith("dec_empBig") and not k.startswith("hours")}
        print(f"  {cond:7s} hours {hh}\n          slope {sl} test {tt}")
