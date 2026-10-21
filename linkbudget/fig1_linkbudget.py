"""Fig. 1 of the paper: prints the numbers behind panels A and B (the link budget of a voxel 5 cm from the probe, every Monte
Carlo input at its median, K = 2 one-second images per 2-s scan), the Monte Carlo spread at 5 cm, the noise that one probe adds
to the readouts' voxels and the pessimistic reach (panel C and Section 3.1), checks each against the paper, and draws panel C.

Physics only needs linkbudget/params.yaml. Panel C and the native fMRI SNR need per-voxel data; when these exist in $WORK_DIR
(inputs/<S>/us_maps.npz with the K = 2 keys and anat_cols.npy, geometry/<S>_geometry.npz, <S>_sensor.json,
us_conditions_summary.json; see README) they are recomputed, otherwise the committed summary summary/fig1c_summary.json
(binned quantiles and medians only, no per-voxel values) is used. --write-summary rebuilds that summary from the per-voxel data.
Outputs: $WORK_DIR/linkbudget/fig1_numbers.json, fig1c.png, fig1c.pdf.  CPU only, about 10 s.
Exit status 0; with --strict, 1 if any number differs from the paper.
Usage: python fig1_linkbudget.py [--write-summary] [--strict]"""
import os, sys, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import vvd_paths as VP
import params as PR
import snr_model as SM
import depth_curves as DC

SUBJ = ("S1", "S2", "S3")
SUMMARY = os.path.join(HERE, "summary", "fig1c_summary.json")
EDGES = np.arange(10, 82, 4.0)                                   # 4-mm distance bins, 10-82 mm
CONDS = {"usL10": "pessimistic (p10), one probe", "usL50": "median (p50), one probe", "usL90": "optimistic (p90), one probe",
         "usB90": "optimistic (p90), two probes"}
PAPER = {  # numbers as written in the ISBI 2027 draft (Fig. 1 caption, Sections 2.3, 3.1, 4)
    "echo_below_receiver_noise_dB": "about 10 dB below the receiver noise (Section 3.1)",
    "gap_to_fmri_dB": "22 to 24 dB below fMRI's (Section 3.1)",
    "target_5cm_dB": "-41.4 dB (Section 4)",
    "mc_5cm_p10_p50_p90_dB": "-61.0 / -41.9 / -28.4 dB (Fig. 1 figure data; not in the text)",
    "added_noise_optimistic_dB": "11 to 14 dB (Section 3.1)",
    "added_noise_median_dB": "23 to 26 dB (Section 3.1)",
    "pessimistic_reach_cm": "1.9 to 4.9 cm (Section 3.1)",
    "dtilde_vs_dprime_dB": "13.5 dB below d' (rho sqrt(17.8/27) ~ 0.21) (Section 2.3)",
}


def echo_budget(zc, P):
    """two-way echo budget at depth zc (cm): the thermal-noise SNR of the echo table split into its components"""
    f = float(PR.F_MHZ); T = PR.T_IMAGE
    sn_tab = float(DC.lin_ext(zc, DC.SN))                                    # table s_n: 30-mm aperture, 2.7-mm bone
    bone_abs_ref = 2 * PR.BONE_COEF * f ** PR.BONE_EXP * PR.BONE_REF_MM; interf = 2 * PR.BONE_SURF_DB
    scalp = 2 * PR.SCALP_DB_CM_MHZ * f * PR.SCALP_CM
    brain = 2 * PR.BRAIN_COEF * f ** PR.BRAIN_EXP * (zc - PR.SCALP_CM - PR.BONE_REF_MM / 10)   # rest of the path is brain/CSF
    aberr = PR.ABERR_DB
    unatt = sn_tab + bone_abs_ref + interf + scalp + brain + aberr           # echo with no skull or tissue loss (back-computed)
    dbone = 2 * PR.BONE_COEF * f ** PR.BONE_EXP * (P["bone_mm"] - PR.BONE_REF_MM)
    sn = sn_tab - PR.AP20_SN_LOSS_DB - dbone + P["cal_db"]
    s_tc = 10 ** ((SM.S_TC_2MHZ + P["leak_db"]) / 10); s_rev = 10 ** ((float(DC.lin_ext(zc, DC.REV)) - dbone + P["leak_db"]) / 10)
    sn_l = 10 ** (sn / 10); s = 1 / (1 / sn_l + 1 / s_tc + 1 / s_rev)
    clutter = -10 * np.log10(1 + sn_l / s_tc); rev_only = -10 * np.log10(1 / sn_l + 1 / s_tc + 1 / s_rev) - (sn + clutter)
    rows = [("Echo with no skull or tissue loss (back-computed)", unatt), (f"Bone absorption ({P['bone_mm']:.1f} mm, two-way)", -(bone_abs_ref + dbone)),
            ("Bone surfaces (4 crossings)", -interf), (f"Scalp and muscle ({10 * PR.SCALP_CM:.0f} mm)", -scalp),
            (f"Brain tissue ({zc - PR.SCALP_CM - PR.BONE_REF_MM / 10:.1f} cm, two-way)", -brain), ("Skull aberration (corrected)", -aberr),
            (f"{PR.APERTURE_MM} mm aperture (not 30 mm)", -PR.AP20_SN_LOSS_DB), ("Calibration (median: 0 dB)", P["cal_db"]),
            ("Clutter + reverberation leakage", clutter + rev_only)]
    tot = sum(v for _, v in rows); assert abs(tot - 10 * np.log10(s)) < 1e-9, (tot, 10 * np.log10(s))
    M = 10 ** (float(DC.lin_ext(zc, DC.MDB)) / 10) / PR.AP20_CELLS_DIV
    return {"rows": rows, "s_n_dB": sn, "s_dB": float(10 * np.log10(s)), "s": float(s), "M": M, "n_ind": M * T / SM.tau_c(f, P["v"])}


def functional_budget(e, P, K):
    s, n_ind = e["s"], e["n_ind"]; snr0 = np.sqrt(n_ind) * s / (1 + s)
    fn = [("Echo SNR, counted twice (below the knee)", 20 * np.log10(s)), ("Saturation near the knee, s/(1+s)", -20 * np.log10(1 + s)),
          (f"{n_ind:.0f} independent samples per 1-s image", 10 * np.log10(n_ind)), (f"{K} images per 2-s scan", 10 * np.log10(K)),
          (f"Response size delta: {100 * P['delta']:.1f} % blood-volume change", 20 * np.log10(P["delta"])),
          (f"Physiological noise (lambda = {P['lam']:.2f})", -10 * np.log10(1 + P["lam"] ** 2 * K * snr0 ** 2))]
    dp = sum(v for _, v in fn)
    dpm = float(20 * np.log10(DC.model_z(PR.Z_REF_CM, K=K, delta=P["delta"], lam=P["lam"], v=P["v"], cal_db=P["cal_db"], bone_mm=P["bone_mm"], leak_db=P["leak_db"])))
    assert abs(dp - dpm) < 1e-9, (dp, dpm)
    return fn, dpm


def per_voxel_available():
    need = [f"{VP.INPUTS}/{s}/{f}" for s in SUBJ for f in ("us_maps.npz", "anat_cols.npy")] + \
           [f"{VP.WORK}/geometry/{s}_{f}" for s in SUBJ for f in ("geometry.npz", "sensor.json")] + [f"{VP.WORK}/geometry/us_conditions_summary.json"]
    return all(os.path.exists(p) for p in need)


def panel_c_from_maps(p10_curve, rho_db):
    """noise added, 10 log10(1 + f_v), to the readouts' (anatomical) voxels per 4-mm distance bin; usL10 (pessimistic) is built here
    with the rules of k2_maps.py, the other conditions are the K = 2 keys of us_maps.npz"""
    out = {"subjects": {}, "bins_mm": EDGES.tolist()}; geo = json.load(open(f"{VP.WORK}/geometry/us_conditions_summary.json"))
    for subj in SUBJ:
        z = np.load(f"{VP.INPUTS}/{subj}/us_maps.npz"); anat = np.load(f"{VP.INPUTS}/{subj}/anat_cols.npy"); r = z["r_eb"].astype(float)
        g = np.load(f"{VP.WORK}/geometry/{subj}_geometry.npz"); vmm3 = json.load(open(f"{VP.WORK}/geometry/{subj}_sensor.json"))["voxel_mm3"]
        d = g["d_left_mm"].astype(float); valid = (d >= PR.DMIN) & (d <= PR.DMAX) & np.isfinite(d)
        tdb = np.interp(d / 10.0, DC.Z_GRID, p10_curve) + rho_db + 10 * np.log10(vmm3 / PR.V_LB); c10 = np.where(valid & (tdb >= PR.FLOOR_DB))[0]
        maps = {"usL10": (c10, np.maximum(0.0, r / 10 ** (tdb / 10) - 1.0)[c10], d[c10])}
        for c, k in (("usL90", "usL90k2"), ("usL50", "usL50k2"), ("usB90", "usB90k2")):
            maps[c] = (z[k + "|cols"], z[k + "|f"].astype(float), z[k + "|d_mm"].astype(float))
        so = {"n_anat": int(anat.size), "native_median_eb_dB": geo[subj]["native"]["median_ncsnr_eb_dB"]}
        for c, (cols, ff, dd0) in maps.items():
            m = np.isin(cols, anat); loss = 10 * np.log10(1 + ff[m]); dd = dd0[m]; q = []
            for a in EDGES[:-1]:
                k = (dd >= a) & (dd < a + 4)
                q.append([float(np.percentile(loss[k], pp)) if k.sum() >= 20 else None for pp in (10, 50, 90)] + [int(k.sum())])
            so[c] = {"n_kept": int(m.sum()), "loss_median": float(np.median(loss)), "binned": q, "d_range_mm": [float(dd.min()), float(dd.max())]}
        out["subjects"][subj] = so
    return out


def draw_panel_c(C, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    ctr = (EDGES[:-1] + 2) / 10; fig, ax = plt.subplots(figsize=(3.4, 1.9))
    for r in (-6, -12, -18):
        ax.axhline(r, ls=":", color="0.55", lw=0.6); ax.text(8.65, r, f"{r} dB", va="center", fontsize=6, color="0.4")
    for c, col, band, dashed in (("usL10", "0.62", False, True), ("usL50", "0.10", True, False), ("usL90", "0.50", True, False)):
        Q = np.array([[[np.nan if x is None else x for x in b[:3]] for b in C["subjects"][s][c]["binned"]] for s in SUBJ], float)   # (subj, bin, 3)
        ok = ~np.isnan(Q[:, :, 1]).any(0); q = -Q.mean(0)
        if band: ax.fill_between(ctr[ok], q[ok, 0], q[ok, 2], color=col, alpha=0.18, lw=0)
        ax.plot(ctr[ok], q[ok, 1], "--" if dashed else "-", color=col, lw=1.0 if dashed else 1.3, label=CONDS[c].split(",")[0])
    ax.set_xlim(0.6, 8.6); ax.set_xticks(range(1, 9)); ax.set_ylim(-54, 0); ax.set_yticks(range(-48, 1, 12))
    ax.set_xlabel("Distance from the probe (cm)"); ax.set_ylabel("SNR change (dB)\nmean of 3 participants")
    ax.set_title("SNR change of the readouts' voxels, one probe", fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="lower left"); fig.tight_layout()
    fig.savefig(path + ".png", dpi=300); fig.savefig(path + ".pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write-summary", action="store_true"); ap.add_argument("--strict", action="store_true"); a = ap.parse_args()
    OUT = os.path.join(VP.WORK, "linkbudget"); os.makedirs(OUT, exist_ok=True)
    K = PR.K_SCAN; P = PR.analytic_medians(); zc = PR.Z_REF_CM
    rho_db = 20 * np.log10(PR.RHO); vox_db = 10 * np.log10(PR.V_FMRI / PR.V_LB)
    e = echo_budget(zc, P); fn, dpm = functional_budget(e, P, K); target = dpm + rho_db + vox_db
    S = DC.draws(); N = len(S["z"]); other = {k: v for k, v in S.items() if k != "z"}
    mc5 = 20 * np.log10(DC.model_z(np.full(N, zc), K=K, **other)) + rho_db + vox_db
    mcq = [float(np.percentile(mc5, p)) for p in (10, 50, 90)]
    p10 = np.array([float(np.percentile(20 * np.log10(DC.model_z(np.full(N, z), K=K, **other)), 10)) for z in DC.Z_GRID])
    tp10 = p10 + rho_db + vox_db; okz = (DC.Z_GRID >= PR.DMIN / 10) & (DC.Z_GRID <= PR.DMAX / 10) & (tp10 >= PR.FLOOR_DB)
    reach = [float(DC.Z_GRID[okz].min()), float(DC.Z_GRID[okz].max())]
    if per_voxel_available():
        C = panel_c_from_maps(p10, rho_db); src = "recomputed from the per-voxel data in $WORK_DIR"
        if a.write_summary:
            os.makedirs(os.path.dirname(SUMMARY), exist_ok=True)
            json.dump({"note": "Fig. 1C summary: noise added, 10 log10(1 + f_v), to the anatomical (readouts') voxels kept under each condition; "
                               "per 4-mm distance bin [p10, p50, p90, n] (quantiles only where n >= 20), per participant; native fMRI median "
                               "per-voxel SNR (empirical Bayes, all cortex). Written by fig1_linkbudget.py --write-summary.", **C}, open(SUMMARY, "w"), indent=1)
            print("summary written:", SUMMARY)
    else:
        if a.write_summary: sys.exit("--write-summary needs the per-voxel data in $WORK_DIR (see README)")
        C = json.load(open(SUMMARY)); src = "from the committed summary " + os.path.relpath(SUMMARY, os.path.dirname(HERE))
    nat = [C["subjects"][s]["native_median_eb_dB"] for s in SUBJ]
    gap = [n - target for n in nat]
    med = {c: [C["subjects"][s][c]["loss_median"] for s in SUBJ] for c in CONDS}

    line = "-" * 100
    print(line); print(f"Fig. 1A: two-way echo budget, voxel {zc:.0f} cm from the probe, {PR.F_MHZ} MHz, Monte Carlo inputs at their medians"); print(line)
    tot = 0.0
    for lab, v in e["rows"]:
        tot += v; print(f"  {lab:58s} {v:+8.2f} dB   (running {tot:+7.2f})")
    print(f"  {'Echo SNR per Doppler sample s':58s} {e['s_dB']:+8.2f} dB   (thermal only s_n {e['s_n_dB']:+.2f} dB)")
    print(line); print(f"Fig. 1B: speech signal in one 2-s scan (K = {K} one-second images), same voxel"); print(line)
    tot = 0.0
    for lab, v in fn:
        tot += v; print(f"  {lab:58s} {v:+8.2f} dB   (running {tot:+7.2f})")
    print(f"  {'d-prime, 27 mm^3 voxel, one 2-s scan':58s} {dpm:+8.2f} dB")
    print(f"  {'Stimulus-specific share (rho = %.2f)' % PR.RHO:58s} {rho_db:+8.2f} dB")
    print(f"  {'27 to %.1f mm^3 voxel (fMRI grid)' % PR.V_FMRI:58s} {vox_db:+8.2f} dB")
    print(f"  {'Ultrasound target d~ per 2-s scan':58s} {target:+8.2f} dB   (d~ = {10 ** ((rho_db + vox_db) / 20):.3f} d')")
    print(f"  fMRI native median per-voxel SNR S1/S2/S3: {' / '.join(f'{x:.2f}' for x in nat)} dB  ->  gap {' / '.join(f'{x:.1f}' for x in gap)} dB")
    print(line); print("Monte Carlo (20,000 draws, seed 0) and panel C   [" + src + "]"); print(line)
    print(f"  target at 5 cm, p10 / p50 / p90: {mcq[0]:.1f} / {mcq[1]:.1f} / {mcq[2]:.1f} dB")
    print(f"  pessimistic (p10) voxels above the {PR.FLOOR_DB:.0f} dB floor: {reach[0]:.2f} to {reach[1]:.2f} cm from the probe")
    for c, lab in CONDS.items():
        print(f"  median noise added to the readouts' voxels, {lab:32s}: {' / '.join(f'{x:5.2f}' for x in med[c])} dB "
              f"(kept {' / '.join(str(C['subjects'][s][c]['n_kept']) for s in SUBJ)} of {' / '.join(str(C['subjects'][s]['n_anat']) for s in SUBJ)} voxels)")

    from decimal import Decimal, ROUND_HALF_UP                              # round half up, as in the text (1.85 -> 1.9)
    r1 = lambda x: float(Decimal(f"{x:.6f}").quantize(Decimal("0.1"), ROUND_HALF_UP)); r0 = lambda x: int(Decimal(f"{x:.6f}").quantize(Decimal("1"), ROUND_HALF_UP))
    checks = [("echo_below_receiver_noise_dB", f"{-e['s_dB']:.1f} dB below (s = {e['s_dB']:.2f} dB)", abs(-e["s_dB"] - 10) < 0.5),
              ("gap_to_fmri_dB", f"{min(gap):.1f} to {max(gap):.1f} dB", (r0(min(gap)), r0(max(gap))) == (22, 24)),
              ("target_5cm_dB", f"{target:.2f} dB", r1(target) == -41.4),
              ("mc_5cm_p10_p50_p90_dB", " / ".join(f"{x:.1f}" for x in mcq), [r1(x) for x in mcq] == [-61.0, -41.9, -28.4]),
              ("added_noise_optimistic_dB", f"{min(med['usL90']):.2f} to {max(med['usL90']):.2f} dB", (r0(min(med["usL90"])), r0(max(med["usL90"]))) == (11, 14)),
              ("added_noise_median_dB", f"{min(med['usL50']):.2f} to {max(med['usL50']):.2f} dB", (r0(min(med["usL50"])), r0(max(med["usL50"]))) == (23, 26)),
              ("pessimistic_reach_cm", f"{reach[0]:.2f} to {reach[1]:.2f} cm", (r1(reach[0]), r1(reach[1])) == (1.9, 4.9)),
              ("dtilde_vs_dprime_dB", f"{-(rho_db + vox_db):.2f} dB below d'", r1(-(rho_db + vox_db)) == 13.5)]
    print(line); print("Check against the paper (rounded as in the text)"); print(line)
    for k, got, ok in checks:
        print(f"  {'MATCH   ' if ok else 'MISMATCH'} {k:30s} code: {got:34s} paper: {PAPER[k]}")
    draw_panel_c(C, os.path.join(OUT, "fig1c"))
    json.dump({"echo_rows_dB": e["rows"], "functional_rows_dB": fn, "dprime_dB": dpm, "target_5cm_dB": target, "native_median_dB": nat, "gap_dB": gap,
               "mc_5cm_p10_p50_p90_dB": mcq, "pessimistic_reach_cm": reach, "added_noise_median_dB": med, "source_panel_c": src,
               "checks": {k: ok for k, _, ok in checks}}, open(os.path.join(OUT, "fig1_numbers.json"), "w"), indent=1)
    print(f"  {sum(ok for _, _, ok in checks)} of {len(checks)} numbers match the paper")
    print(line); print(f"wrote {OUT}/fig1_numbers.json, fig1c.png, fig1c.pdf"); print(line)
    return all(ok for _, _, ok in checks) or not a.strict


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
