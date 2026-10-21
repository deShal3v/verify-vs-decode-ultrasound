"""Figure data from the K = 2 grid (GRID_E4_s7.json, verifiers/analysis/grid.py), same keys as the earlier K = 10 figure data, so the
figure scripts draw them unchanged: usL90, usB90, usL50 and the uniform gains hold the K = 2 conditions (usL90k2, usB90k2, usL50k2,
usL50k2u6/u12/u18); m18c now carries the decoder at x63 (Phase E4b), and the noise-survived ladder includes it.
Writes <out>/fig2_data_e4.json and <out>/fig3_e4_data.json (the figure scripts' inputs).
Run: python make_fig_data.py results/GRID_E4_s7.json figures/figdata"""
import os, sys, json, numpy as np
G = json.load(open(sys.argv[1])); FD = sys.argv[2]; os.makedirs(FD, exist_ok=True)
num = lambda v: float(v) if isinstance(v, (int, float)) else (np.inf if v == "+inf" else -np.inf if v == "-inf" else np.nan)
MAP = {"native": "native", "m6c": "m6c", "m12c": "m12c", "m18c": "m18c", "usL90": "usL90k2", "usB90": "usB90k2", "usL50": "usL50k2"}
NAMES = ["ver_C", "ver_ETC", "dec_empBig", "dec_gaussBig"]; RUNGS = ["n1", "n2", "n4", "n8", "n16", "n32", "n48", "n56", "n64", "n73", "n82"]
def pk(iv, hours=True):
    lo, hi = num(iv[0]), num(iv[1])
    if hours and lo == 0: lo = -np.inf
    if np.isfinite(lo) and np.isfinite(hi): return lo, "point"
    if np.isfinite(hi): return hi, "below"
    if np.isfinite(lo): return lo, "above"
    return np.nan, "none"
OUT = {"subjects": list(G["per_subject"]), "conds": list(MAP), "source": "GRID_E4_s7.json (K = 2)"}; F3 = {}
for S, O in G["per_subject"].items():
    o = {"bar": O["bar"]}
    for c, src in MAP.items():
        e = O["conditions"].get(src, {})
        for n in NAMES:
            cur = e.get("curves", {}).get(n, []); o[f"cur_{c}_{n}"] = [[p[1], p[2]] for p in cur]
            sl = e.get("slope", {}).get(n); o[f"slope_{c}_{n}"] = [sl["s"], sl["ci95"][0], sl["ci95"][1]] if sl else [None, None, None]
            cr = e.get("cross", {}).get(n); v, k = pk(cr["hours_interval"]) if cr else (np.nan, "none"); o[f"hours_{c}_{n}"] = v; o[f"hours_{c}_{n}_kind"] = k
    for n in NAMES:
        vals, kinds = [], []
        for r in RUNGS:
            x = O["noise_survived"].get(r, {}).get(n)
            if not x: vals.append(np.nan); kinds.append("none"); continue
            v, k = pk(x["interval"], hours=False); vals.append(v); kinds.append(k)
        o[f"ns_{n}"] = vals; o[f"ns_{n}_kind"] = kinds
    o["rung_hours"] = [next((p[1] for c in MAP.values() for p in O["conditions"].get(c, {}).get("curves", {}).get("ver_C", []) if p[0] == r), None) for r in RUNGS]
    OUT[S] = o
for key, src in (("usL90", "usL90k2"), ("usL50", "usL50k2"), ("usL50u6", "usL50k2u6"), ("usL50u12", "usL50k2u12"), ("usL50u18", "usL50k2u18")):
    bits, hours = [], []
    for S in ("S1", "S2", "S3"):
        e = G["per_subject"][S]["conditions"].get(src, {}); cur = {p[0]: p[2] for p in e.get("curves", {}).get("ver_C", [])}
        bits.append(cur.get("n82")); hours.append(e.get("cross", {}).get("ver_C", {}).get("hours_interval"))
    F3[key] = {"bits": bits, "hours": hours, "source": src}
def nan2none(x):
    if isinstance(x, dict): return {k: nan2none(v) for k, v in x.items()}
    if isinstance(x, list): return [nan2none(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x): return None
    return x
json.dump(nan2none(OUT), open(os.path.join(FD, "fig2_data_e4.json"), "w"), indent=1)
json.dump(nan2none(F3), open(os.path.join(FD, "fig3_e4_data.json"), "w"), indent=1); print("fig2_data_e4.json, fig3_e4_data.json written")
