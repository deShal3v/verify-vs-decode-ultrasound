"""Depth curves d'_p(z) of the temple link budget (snr_model.py) with depth as a continuous input.

Depth extension: the echo table gives, at 2 MHz, rows z = 2, 3, 4, 5 cm. Each depth-dependent quantity is interpolated
piecewise-linearly in dB per cm between rows and extrapolated linearly in dB per cm with the slope of the nearest end segment:
  s_n(z)   : +3, 0, -3, -6 dB  -> exactly -3 dB/cm (9 - 3z)
  M(z)     : 89, 39, 22, 14 cells per 27 mm^3 -> 10log10 M piecewise linear (-3.58, -2.49, -1.96 dB/cm)
  s_rev(z) : -10, 0, +10, +20 dB -> exactly +10 dB/cm
Everything else is snr_model.model() unchanged (20 mm aperture: -2 dB and M/2.25; bone loss 2*2.1*2^1.1*(bone-2.7);
reverberation degraded by the extra bone loss; tissue clutter +5 dB; voxel 27 mm^3).
Monte Carlo: the six non-depth inputs drawn exactly as snr_montecarlo.py (same distributions, N = 20,000, seed 0,
same draw order with the depth draw kept in place so the other draws are identical); for every depth on the grid all
20,000 draws are evaluated (common random numbers) and the 10th / 50th / 90th percentiles of 20log10 d' are taken.
Checks: (1) the continuous model equals snr_model.model() at z = 2, 3, 4, 5 for every 50th draw; (2) re-running the original
Monte Carlo (z drawn from {3,4,5}) reproduces snr_montecarlo.py.
The curves written here are per 10-s trial (K = 10, the function default; the form used before the K = 2 per-scan budget);
k2_maps.py and fig1_linkbudget.py evaluate model_z(K = 2) on the same draws.
Outputs: $WORK_DIR/linkbudget/depth_curves.json, depth_curves.png.  Usage: python depth_curves.py"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import params as PR
import snr_model as SM
ZT = np.array([2.0, 3.0, 4.0, 5.0])
SN = np.array([SM.TABLE[(2, z)][0] for z in (2, 3, 4, 5)], float)
MDB = 10 * np.log10(np.array([SM.TABLE[(2, z)][1] for z in (2, 3, 4, 5)], float))
REV = np.array([SM.S_REV_2MHZ[z] for z in (2, 3, 4, 5)], float)
Z_GRID = np.round(np.arange(*PR.Z_GRID_SPEC), 4)

def lin_ext(z, y, x=ZT):
    """piecewise-linear interpolation, linear extrapolation with the end-segment slopes"""
    z = np.asarray(z, float); out = np.interp(z, x, y)
    lo = z < x[0]; hi = z > x[-1]
    out = np.where(lo, y[0] + (z - x[0]) * (y[1] - y[0]) / (x[1] - x[0]), out)
    out = np.where(hi, y[-1] + (z - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]), out)
    return out

def model_z(z, delta=PR.DEF["delta"], lam=PR.DEF["lam"], K=PR.DEF["K"], T=PR.T_IMAGE, v=PR.DEF["v_mm_per_s"], aperture_mm=PR.APERTURE_MM,
            cal_db=0.0, bone_mm=PR.BONE_REF_MM, leak_db=0.0, voxel_mm3=PR.V_LB, m_law="db_per_cm"):
    """snr_model.model(f=2, ...) with continuous depth z (cm); vectorised over all arguments"""
    db = lambda x: 10 ** (np.asarray(x) / 10)
    sn = lin_ext(z, SN) + cal_db
    M = 10 ** (lin_ext(z, MDB) / 10) if m_law == "db_per_cm" else 356.0 / np.asarray(z, float) ** 2
    if aperture_mm == 20: sn = sn - PR.AP20_SN_LOSS_DB; M = M / PR.AP20_CELLS_DIV
    M = M * voxel_mm3 / PR.V_LB
    dbone = 2 * PR.BONE_COEF * 2 ** PR.BONE_EXP * (np.asarray(bone_mm) - PR.BONE_REF_MM)
    sn = sn - dbone
    L = 1 / db(SM.S_TC_2MHZ + leak_db) + 1 / db(lin_ext(z, REV) - dbone + leak_db)
    s = 1 / (1 / db(sn) + L)
    return SM.dprime(s, M, 2, delta, lam, K, T, v)

def draws(N=PR.MC_N, seed=PR.MC_SEED):
    """exactly snr_montecarlo.py's draws (same order), including its depth draw"""
    rng = np.random.default_rng(seed); R = PR.MC_RANGES
    logu = lambda a, b, n: np.exp(rng.uniform(np.log(a), np.log(b), n))
    S = dict(cal_db=rng.uniform(*R["cal_db"][1], N), delta=logu(*R["delta"][1], N), bone_mm=rng.uniform(*R["bone_mm"][1], N), z=rng.choice(PR.MC_Z_CHOICE, N),
             leak_db=rng.uniform(*R["leak_db"][1], N), lam=logu(*R["lam"][1], N), v=logu(*R["v"][1], N))
    return S

def main():
    import vvd_paths as VP
    import snr_montecarlo as MC
    OUT = os.path.join(VP.WORK, "linkbudget"); os.makedirs(OUT, exist_ok=True)
    S = draws(); N = len(S["z"]); other = {k: v for k, v in S.items() if k != "z"}
    # check 1: continuous model == table model at the table depths
    for z in (2, 3, 4, 5):
        a = model_z(np.full(N, float(z)), **other)
        b = np.array([SM.model(f=2, z=z, delta=S["delta"][i], lam=S["lam"][i], v=S["v"][i], cal_db=S["cal_db"][i], bone_mm=S["bone_mm"][i], leak_db=S["leak_db"][i]) for i in range(0, N, 50)])
        assert np.allclose(a[::50], b, rtol=1e-10), (z, np.abs(a[::50] - b).max())
    # check 2: original Monte Carlo reproduced
    d0 = 20 * np.log10(model_z(S["z"].astype(float), **other))
    ref = MC.run()["pct10_25_50_75_90_dB"]
    q0 = np.percentile(d0, [10, 25, 50, 75, 90]); assert np.allclose(q0, ref, atol=1e-9), (q0, ref)
    P = {10: [], 50: [], 90: []}; Pp = {10: [], 50: [], 90: []}
    for z in Z_GRID:
        d = 20 * np.log10(model_z(np.full(N, z), **other)); q = np.percentile(d, [10, 50, 90])
        dp = 20 * np.log10(model_z(np.full(N, z), m_law="power", **other)); qp = np.percentile(dp, [10, 50, 90])
        for i, p in enumerate((10, 50, 90)): P[p].append(float(q[i])); Pp[p].append(float(qp[i]))
    base = [float(20 * np.log10(model_z(z))) for z in Z_GRID]
    out = {"z_cm": Z_GRID.tolist(), "dprime_dB": {str(p): P[p] for p in P}, "baseline_dB": base,
           "sensitivity_M_power_law_dB": {str(p): Pp[p] for p in Pp},
           "table_depths_cm": ZT.tolist(), "extension": "piecewise linear in dB per cm between TABLE rows, end-segment slopes outside 2-5 cm",
           "slopes_dB_per_cm": {"s_n": -3.0, "M_10log10": [float(x) for x in np.diff(MDB)], "s_rev": 10.0},
           "N": int(N), "checks": {"table_rows_equal_snr_model": True, "original_mc_reproduced": [float(x) for x in q0]},
           "units": "20log10 d' per 27 mm^3 voxel, one 10 s trial, 2 MHz, 20 mm aperture (snr_model defaults)"}
    for p in (10, 50, 90):
        for zz in (2, 3, 4, 5, 6, 7, 8):
            out.setdefault("at_cm", {}).setdefault(str(zz), {})[str(p)] = float(np.interp(zz, Z_GRID, P[p]))
    out["max_dev_power_law_dB_2_to_8cm"] = float(max(np.max(np.abs(np.array(P[p]) - np.array(Pp[p]))[(Z_GRID >= 2) & (Z_GRID <= 8)]) for p in P))
    json.dump(out, open(f"{OUT}/depth_curves.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for p, c in ((90, "#2a78d6"), (50, "#eb6834"), (10, "#52514e")):
        ax.plot(Z_GRID, P[p], color=c, lw=1.6, label=f"{p}th percentile")
        ax.plot(Z_GRID, Pp[p], color=c, lw=0.8, ls=":")
    ax.plot(Z_GRID, base, color="k", lw=0.8, ls="--", label="baseline inputs")
    ax.axvspan(2, 5, color="0.92", zorder=0); ax.text(3.5, ax.get_ylim()[1] - 2, "TABLE rows", ha="center", va="top", fontsize=7)
    ax.set_xlabel("Depth from the skin z (cm)"); ax.set_ylabel("20 log10 d' (dB)\n27 mm³ voxel, 10 s trial")
    ax.set_title("Temple link budget vs depth, 6 other inputs drawn (N = 20,000); dotted: M ∝ z⁻²", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/depth_curves.png", dpi=200); plt.close(fig)
    print(json.dumps({k: out[k] for k in ("at_cm", "max_dev_power_law_dB_2_to_8cm", "checks")}, indent=1))

if __name__ == "__main__":
    main()
