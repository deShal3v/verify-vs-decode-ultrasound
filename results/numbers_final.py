"""Every result number the ISBI text uses, printed from the reconciled run (GRID_E4_s7.json, verifiers/analysis/grid.py: Phase E + E2 + E3 + E4 merged;
uniform noise from Phase E/E2, ultrasound conditions at K = 2 from E4/E4c). Per participant S1, S2, S3; ranges are min to max.
Noise survived in dB of SNR change = 10 log10 of the noise-power multiple (negative = more noise).
Run: python results/numbers_final.py [results/GRID_E4_s7.json]   (default: the grid file committed next to this script)"""
import os, sys, json, numpy as np
GP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "GRID_E4_s7.json")
G = json.load(open(GP))["per_subject"]; SS = ["S1", "S2", "S3"]
f = lambda v: "None" if v is None else (f"{v:.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else str(v))
def cond(S, c): return G[S]["conditions"].get(c, {})
def bits(S, c, r, rung="n82"):
    return next((p[2] for p in cond(S, c).get("curves", {}).get(r, []) if p[0] == rung), None)
def hours(S, c, r):
    x = cond(S, c).get("cross", {}).get(r); return None if not x else x["hours_interval"]
def slope(S, c, r):
    x = cond(S, c).get("slope", {}).get(r); return None if not x else (round(x["s"], 2), [round(v, 2) for v in x["ci95"]])
def tests(S, c):
    return {k: (round(v["d"], 2), [round(x, 2) for x in v["ci95"]], v["verdict"]) for k, v in cond(S, c).get("tests", {}).items() if not k.startswith("hours")}
print("bar (decoder goal):", [round(G[S]["bar"], 2) for S in SS])
for c in ("native", "m6c", "m12c", "m18c", "usL90k2", "usB90k2", "usL50k2", "usL50k2u6", "usL50k2u12", "usL50k2u18"):
    print(f"\n== {c}")
    for r in ("ver_C", "ver_ETC", "dec_empBig"):
        print(f"  {r:10s} bits15h", [f(bits(S, c, r)) for S in SS], " hours", [hours(S, c, r) for S in SS], " slope", [slope(S, c, r) for S in SS])
    for S in SS:
        t = tests(S, c)
        if t: print("  tests", S, t)
print("\n== noise survived at n82 (dB of SNR change, kind)")
for r in ("ver_C", "ver_ETC", "dec_empBig"):
    v = [G[S]["noise_survived"].get("n82", {}).get(r) for S in SS]
    print(f"  {r:10s}", [(None if not x else [None if y is None else (round(float(y), 2) if not isinstance(y, str) else y) for y in x["interval"]]) for x in v])
print("\n== slope change native -> m6c (bits per tenfold)")
for r in ("ver_C", "ver_ETC", "dec_empBig"):
    print(f"  {r:10s}", [round(slope(S, "m6c", r)[0] - slope(S, "native", r)[0], 2) for S in SS])
