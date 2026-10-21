"""Phase C per-voxel ultrasound conditions (DESIGN_C.md section 4): empirical-Bayes native SNR, target SNR from the
depth curves, added-variance factor f_v, reachable voxel sets, area counts, figures.
Inputs ($WORK_DIR): geometry/ncsnr_<S>.npz (ncsnr_stream.py), geometry/<S>_geometry.npz (linkbudget/build_geometry.py),
linkbudget/depth_curves.json (linkbudget/depth_curves.py), <S>/areas.json (voxels/atlas_sets.py).
rho, TR, the -60 dB floor and the 1-8 cm validity come from linkbudget/params.yaml.
Outputs ($WORK_DIR/geometry): <S>_us_maps.npz (input of linkbudget/k2_maps.py), us_conditions_summary.json, <S>_snr_maps.png, eb_check.png
Usage: python snr_maps.py"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "linkbudget"))
import params as PR
OUT = f"{VP.WORK}/geometry"
AREAS = {s: f"{VP.WORK}/{s}/areas.json" for s in ("S1", "S2", "S3")}
RHO = PR.RHO; TR = PR.TR; FLOOR_DB = PR.FLOOR_DB; DMIN, DMAX = PR.DMIN, PR.DMAX
CONDS = {  # name: (distance key, percentile, rho, native estimator)
    "usL10": ("d_left_mm", "10", RHO, "eb"), "usL50": ("d_left_mm", "50", RHO, "eb"), "usL90": ("d_left_mm", "90", RHO, "eb"),
    "usB50": ("d_both_mm", "50", RHO, "eb"), "usB90": ("d_both_mm", "90", RHO, "eb"),
    "usL50lo": ("d_left_mm", "50", 0.21, "eb"), "usL50hi": ("d_left_mm", "50", 0.31, "eb"), "usL50raw": ("d_left_mm", "50", RHO, "raw"),
    "usL50cone": ("d_left_cone27_mm", "50", RHO, "eb"), "usL90cone": ("d_left_cone27_mm", "90", RHO, "eb")}


def npmle_posterior_mean(r, se, n_grid=400, iters=500, tol=1e-9):
    """EB posterior mean of r >= 0 given r_hat ~ N(r, se^2); prior = NPMLE on a grid (EM)"""
    se = np.maximum(se, 2e-3); rmax = float(np.percentile(r + 3 * se, 99.99))
    grid = np.unique(np.concatenate([np.linspace(0, 0.1, n_grid // 2), np.geomspace(0.1, max(rmax, 0.2), n_grid // 2)]))
    L = np.exp(-0.5 * ((r[:, None] - grid[None]) / se[:, None]) ** 2) / se[:, None]          # (V, G)
    w = np.full(len(grid), 1.0 / len(grid)); ll_old = -np.inf
    for it in range(iters):
        m = L @ w; ll = float(np.log(np.maximum(m, 1e-300)).sum())
        w = w * (L.T @ (1.0 / np.maximum(m, 1e-300))) / len(r)
        if ll - ll_old < tol * abs(ll): break
        ll_old = ll
    m = L @ w; post = (L * w[None]) @ grid / np.maximum(m, 1e-300)
    return post, grid, w, it


def eb_hetero(r, se, n_grid=400, iters=1000, tol=1e-10, n_bins=40):
    """EB posterior mean of r >= 0 with r-dependent sampling noise: r_hat ~ N(r, s(r)^2), where s(.) is the median
    jackknife SE within quantile bins of r_hat (linear interpolation; constant beyond the end bins). The prior is the
    NPMLE on a grid (EM). Using s(r) instead of each voxel's own noisy SE avoids both over-shrinking high-signal voxels
    (their SE is larger) and the artefacts of conditioning the prior on a noisy SE."""
    q = np.quantile(r, np.linspace(0, 1, n_bins + 1)); b = np.clip(np.searchsorted(q, r, side="right") - 1, 0, n_bins - 1)
    rc = np.array([np.median(r[b == k]) for k in range(n_bins)]); sc = np.array([np.median(se[b == k]) for k in range(n_bins)])
    s_of = lambda x: np.interp(x, rc, sc)
    rmax = float(np.percentile(r, 99.99) + 3 * sc[-1])
    grid = np.unique(np.concatenate([np.linspace(0, 0.1, n_grid // 2), np.geomspace(0.1, max(rmax, 0.2), n_grid // 2)]))
    sg = s_of(grid)
    L = np.exp(-0.5 * ((r[:, None] - grid[None]) / sg[None]) ** 2) / sg[None]
    w = np.full(len(grid), 1.0 / len(grid)); ll_old = -np.inf
    for it in range(iters):
        m = L @ w; ll = float(np.log(np.maximum(m, 1e-300)).sum())
        w = w * (L.T @ (1.0 / np.maximum(m, 1e-300))) / len(r)
        if ll - ll_old < tol * abs(ll): break
        ll_old = ll
    m = L @ w; post = (L * w[None]) @ grid / np.maximum(m, 1e-300)
    diag = {"s_of_r_bins": [[float(x), float(y)] for x, y in zip(rc[::4], sc[::4])], "em_iters": int(it), "prior_mass_r_le_0.005": float(w[grid <= 0.005].sum()),
            "prior_mean": float((w * grid).sum()), "data_mean": float(r.mean())}
    return post, diag


def main():
    dc = json.load(open(f"{VP.WORK}/linkbudget/depth_curves.json")); zc = np.array(dc["z_cm"])
    summary = {"rules": {"rho": RHO, "TR_s": TR, "floor_dB": FLOOR_DB, "validity_mm": [DMIN, DMAX], "conds": {k: list(v) for k, v in CONDS.items()}}}
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    figeb, axeb = plt.subplots(1, 3, figsize=(15, 4.2))
    for si, subj in enumerate(["S1", "S2", "S3"]):
        nc = np.load(f"{OUT}/ncsnr_{subj}.npz"); g = np.load(f"{OUT}/{subj}_geometry.npz")
        A = json.load(open(AREAS[subj])); sets = {k: np.array(v, int) for k, v in A["sets"].items()}
        info = json.load(open(f"{OUT}/{subj}_sensor.json")); vmm3 = info["voxel_mm3"]
        r_hat = nc["r"].astype(np.float64); se = nc["se_jack"].astype(np.float64); var_n = nc["var_n"].astype(np.float64)
        r_raw = nc["ncsnr"].astype(np.float64) ** 2
        r_eb, eb_diag = eb_hetero(r_hat, se)
        V = len(r_hat); lab = g["label"]
        S = {"subject": subj, "V": V, "voxel_mm3": vmm3,
             "native": {"median_ncsnr_raw_dB": float(10 * np.log10(np.median(np.maximum(r_raw, 1e-12)))),
                        "median_ncsnr_eb_dB": float(10 * np.log10(np.median(r_eb))), "p90_ncsnr_eb_dB": float(10 * np.log10(np.percentile(r_eb, 90))),
                        "median_r_hat": float(np.median(r_hat)), "median_se_jack": float(np.median(se)), "frac_r_hat_le_0": float(np.mean(r_hat <= 0)),
                        "sum_r_eb": float(r_eb.sum()), "sum_r_raw": float(r_raw.sum()), "sum_r_hat": float(r_hat.sum()), "eb_bins": eb_diag}, "conds": {}}
        maps = {"var_n": var_n.astype(np.float32), "r_eb": r_eb.astype(np.float32), "r_raw": r_raw.astype(np.float32), "r_hat": r_hat.astype(np.float32)}
        for cname, (dkey, p, rho, est) in CONDS.items():
            d = g[dkey].astype(np.float64); z = d / 10.0
            dprime_db = np.interp(z, zc, np.array(dc["dprime_dB"][p]))
            off = 20 * np.log10(rho) + 10 * np.log10(TR / 10.0) + 10 * np.log10(vmm3 / 27.0)
            tdb = dprime_db + off; t2 = 10 ** (tdb / 10)
            inset = (d >= DMIN) & (d <= DMAX) & np.isfinite(d) & (tdb >= FLOOR_DB)
            rn = r_eb if est == "eb" else r_raw
            f = np.maximum(0.0, rn / t2 - 1.0)
            real = np.minimum(rn, t2)                                     # realised SNR^2 (expected)
            cols = np.where(inset)[0]
            maps[f"{cname}|cols"] = cols.astype(np.int32); maps[f"{cname}|f"] = f[cols].astype(np.float32); maps[f"{cname}|target_db"] = tdb[cols].astype(np.float32)
            maps[f"{cname}|d_mm"] = d[cols].astype(np.float32)
            eq = 10 * np.log10(real[cols].sum() / r_eb.sum())
            area = {k: int(np.isin(v, cols).sum()) for k, v in sets.items()}
            area_frac = {k: float(np.isin(v, cols).mean()) for k, v in sets.items()}
            labs, cnt = np.unique(lab[cols], return_counts=True); top = sorted(zip(cnt.tolist(), labs.tolist()), reverse=True)[:12]
            q = lambda x, pp: float(np.percentile(x, pp)) if len(x) else None
            S["conds"][cname] = {
                "n_vox": int(len(cols)), "frac_of_cortex": float(len(cols) / V),
                "n_excluded_shallow_lt_10mm": int((d < DMIN).sum()), "n_excluded_deep_gt_80mm": int(((d > DMAX) | ~np.isfinite(d)).sum()),
                "n_excluded_floor": int(((d >= DMIN) & (d <= DMAX) & (tdb < FLOOR_DB)).sum()),
                "depth_mm": {"min": q(d[cols], 0), "median": q(d[cols], 50), "max": q(d[cols], 100)},
                "target_dB": {"max": q(tdb[cols], 100), "p90": q(tdb[cols], 90), "median": q(tdb[cols], 50), "min": q(tdb[cols], 0)},
                "realised_ncsnr_dB": {"p90": q(10 * np.log10(np.maximum(real[cols], 1e-30)), 90), "median": q(10 * np.log10(np.maximum(real[cols], 1e-30)), 50)},
                "native_ncsnr_dB_in_set": {"median": q(10 * np.log10(np.maximum(rn[cols], 1e-12)), 50), "p90": q(10 * np.log10(np.maximum(rn[cols], 1e-12)), 90)},
                "added_noise_dB": {"median": q(10 * np.log10(1 + f[cols]), 50), "p90": q(10 * np.log10(1 + f[cols]), 90), "p10": q(10 * np.log10(1 + f[cols]), 10)},
                "frac_no_noise_added": float(np.mean(f[cols] == 0)) if len(cols) else None,
                "signal_power_share_of_native_cortex": float(real[cols].sum() / r_eb.sum()), "equivalent_plain_noise_dB": float(eq),
                "area_counts": area, "area_frac_reached": area_frac, "top_desikan_labels": top}
        np.savez_compressed(f"{OUT}/{subj}_us_maps.npz", **maps)
        summary[subj] = S
        # EB check plot
        a = axeb[si]; o = np.argsort(r_hat)
        a.scatter(r_hat[::7], r_eb[::7], s=0.5, alpha=0.4, c="#2a78d6", rasterized=True); a.plot([-0.1, 1], [-0.1, 1], "k:", lw=0.6)
        a.set_xlim(-0.1, 0.5); a.set_ylim(-0.02, 0.5); a.set_xlabel("raw r = ncsnr² (unclipped)"); a.set_ylabel("EB posterior mean")
        a.set_title(f"{subj}: median r {np.median(r_hat):.3f}, median SE {np.median(se):.3f}; sum EB/sum raw {r_eb.sum() / r_hat.sum():.2f}", fontsize=8, loc="left")
        figure(subj, g, maps, S)
        print(subj, json.dumps({c: (v["n_vox"], round(v["target_dB"]["median"], 1) if v["n_vox"] else None, round(v["added_noise_dB"]["median"], 1) if v["n_vox"] else None,
                                    round(v["equivalent_plain_noise_dB"], 1)) for c, v in S["conds"].items()}), flush=True)
    figeb.tight_layout(); figeb.savefig(f"{OUT}/eb_check.png", dpi=120)
    json.dump(summary, open(f"{OUT}/us_conditions_summary.json", "w"), indent=1)


def figure(subj, g, maps, S):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    F = g["frame_xyz"]; Fs = g["sensor_left_frame"]
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    for a, c in zip(ax, ["usL10", "usL50", "usL90", "usB50"]):
        cols = maps[f"{c}|cols"]; tdb = maps[f"{c}|target_db"]
        lh = np.where(g["hemi"] == "lh")[0]
        a.scatter(F[lh, 1], F[lh, 2], s=1, c="0.88", rasterized=True)
        sel = cols[np.isin(cols, lh)] if not c.startswith("usB") else cols
        o = np.argsort(-F[sel, 0]) if not c.startswith("usB") else np.argsort(F[sel, 0])
        sc = a.scatter(F[sel[o], 1], F[sel[o], 2], s=2, c=tdb[np.searchsorted(cols, sel[o])], cmap="magma", vmin=-60, vmax=-20, rasterized=True)
        k = np.arange(0, len(Fs), 5); a.scatter(Fs[k, 1], Fs[k, 2], s=0.3, c="#2a78d6", alpha=0.3)
        s = S["conds"][c]
        a.set_title(f"{subj} {c}: {s['n_vox']} voxels, median target {s['target_dB']['median']:.1f} dB,\nequivalent plain noise {s['equivalent_plain_noise_dB']:.1f} dB (lateral view, lh voxels)", fontsize=8, loc="left")
        a.set_aspect("equal"); a.set_xlabel("frame y (mm)"); a.set_ylabel("frame z (mm)")
    plt.colorbar(sc, ax=ax[-1], label="target SNR per voxel per 2 s scan (dB)")
    fig.tight_layout(); fig.savefig(f"{OUT}/{subj}_snr_maps.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
