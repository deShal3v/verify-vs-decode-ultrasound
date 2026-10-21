"""Phase C geometry, per subject (DESIGN_C.md section 2): temporal-window sensor patch(es) on the scalp, distance from
every functional voxel centre to the closest sensor point, incidence-angle diagnostic, per-voxel Desikan label and
Phase B area-set membership. CPU only.
Inputs: $DATA_DIR/anat/<UTS>/ (geom_lib.py), $WORK_DIR/<S>/areas.json (voxels/atlas_sets.py).
Outputs ($WORK_DIR/geometry/): <S>_geometry.npz, <S>_sensor.json, <S>_sanity.png
Usage: python -I build_geometry.py S1 S2 S3"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geom_lib as G
OUT = f"{VP.WORK}/geometry"; os.makedirs(OUT, exist_ok=True)
AREAS = {s: f"{VP.WORK}/{s}/areas.json" for s in ("S1", "S2", "S3")}


def voxel_labels(S):
    """Desikan label per functional voxel: majority vote over 6 white->pial samples per vertex (Phase B atlas_sets.py)"""
    m = S["mask"]; ijk = S["ijk"]; V = len(ijk); xfm = S["xfm"]
    lut = -np.ones(m.shape, np.int64); lut[tuple(ijk.T)] = np.arange(V)
    names = sorted(set(np.concatenate([S["annot"]["lh"], S["annot"]["rh"]])))
    code = {n: i for i, n in enumerate(names)}
    cnt = np.zeros((V, len(names)), np.int32); hemi_cnt = np.zeros((V, 2), np.int32)
    for hi, h in enumerate(("lh", "rh")):
        wm, pia, _ = S["surf"][h]; lab = np.array([code[x] for x in S["annot"][h]])
        for t in np.linspace(0, 1, 6):
            q = np.rint((wm + t * (pia - wm)) @ xfm[:3, :3].T + xfm[:3, 3]).astype(int)
            ok = (q >= 0).all(1) & (q < np.array(m.shape)).all(1); c = lut[tuple(q[ok].T)]; good = c >= 0
            np.add.at(cnt, (c[good], lab[ok][good]), 1); np.add.at(hemi_cnt, (c[good], hi), 1)
    lab_v = np.where(cnt.sum(1) > 0, np.array(names)[cnt.argmax(1)], "none")
    hemi_v = np.where(hemi_cnt.sum(1) == 0, "none", np.where(hemi_cnt[:, 0] >= hemi_cnt[:, 1], "lh", "rh"))
    return lab_v, hemi_v


def patch(S, side, R, p0, head, v2r):
    """sensor patch (DESIGN_C.md 2.3): lateral head-surface points inside the window box, minus the ear-canal disk.
    Where the auricle lies on the scalp (points > PINNA_EXCESS_MM proud of the robust scalp fit), the sensor point is
    put on the fitted scalp surface underneath (a behind-/under-the-ear housing), with the fitted surface's normal."""
    hemi = "lh" if side == "left" else "rh"
    L = G.landmarks(S, hemi, R, p0)
    P, cells, hh, hmap = G.lateral_surface(head, v2r, side)
    F = G.to_frame(P, R, p0)
    coef, proud, D = G.scalp_fit(F, side, L)
    pin = proud > G.PINNA_EXCESS_MM
    box = G.window_patch(P, F, cells, hh, side, L)
    eam = (F[:, 1] - L["y_EAM"]) ** 2 + (F[:, 2] - L["z_EAM"]) ** 2 <= G.EAM_R_MM ** 2
    sel = box & ~eam
    Fs = F[sel].copy(); ps = pin[sel]
    Fs[ps, 0] = D(Fs[ps, 1], Fs[ps, 2]) @ coef                      # auricle -> fitted scalp beneath it
    Ps = Fs @ R + p0                                                # back to scanner RAS (f = R (p - p0))
    N = G.surface_normals(head, Ps, v2r)
    N[ps] = G.fit_normals(Fs[ps], coef, side) @ R                   # frame -> RAS directions
    resid = proud[box & ~pin]
    return dict(L=L, P_all=P, F_all=F, pin=pin, box=box, eam=eam, sel=sel, P=Ps, F=Fs, N=N, n_pinna_replaced=int(ps.sum()),
                fit_rmse_mm=float(np.sqrt(np.mean(resid ** 2))), fit_coef=coef.tolist())


def dist_and_angle(vox, pts, nrm):
    tree = cKDTree(pts); d, j = tree.query(vox)
    path = vox - pts[j]; path /= np.maximum(np.linalg.norm(path, axis=1, keepdims=True), 1e-9)
    ang = np.degrees(np.arccos(np.clip((path * -nrm[j]).sum(1), -1, 1)))         # 0 deg = straight in along the inward normal
    return d, j, ang


def cone_distance(vox, pts, nrm, max_deg=27.0, ch=1500):
    """distance to the closest sensor point whose inward normal is within max_deg of the path (critical-angle
    sensitivity, DESIGN_C.md 2.6); inf if no sensor point sees the voxel within the cone"""
    c0 = np.cos(np.radians(max_deg)); out = np.full(len(vox), np.inf); inw = -nrm
    for a in range(0, len(vox), ch):
        D = vox[a:a + ch, None, :] - pts[None]; dist = np.linalg.norm(D, axis=2)
        cosang = (D * inw[None]).sum(2) / np.maximum(dist, 1e-9)
        dist = np.where(cosang >= c0, dist, np.inf); out[a:a + ch] = dist.min(1)
    return out


def main(subj):
    S = G.load_subject(subj); R, p0 = G.head_frame(S["tal"]); head = G.head_mask(S["A"]); v2r = S["T1"].header.get_vox2ras()
    vox = S["vox_ras"]; Fv = G.to_frame(vox, R, p0); V = len(vox)
    lab_v, hemi_v = voxel_labels(S)
    pl = patch(S, "left", R, p0, head, v2r); pr = patch(S, "right", R, p0, head, v2r)
    dL, jL, aL = dist_and_angle(vox, pl["P"], pl["N"])
    dR, jR, aR = dist_and_angle(vox, pr["P"], pr["N"])
    dB = np.minimum(dL, dR); aB = np.where(dL <= dR, aL, aR)
    dLc = cone_distance(vox, pl["P"], pl["N"]); dRc = cone_distance(vox, pr["P"], pr["N"]); dBc = np.minimum(dLc, dRc)
    # head surface sanity: every voxel must be inside the head, i.e. farther from outside than 0 (distance to ANY lateral surface point >= ~5 mm)
    allsurf = cKDTree(np.vstack([pl["P_all"], pr["P_all"]])); d_any = allsurf.query(vox)[0]
    A = json.load(open(AREAS[subj])); sets = {k: np.array(v, int) for k, v in A["sets"].items()}
    np.savez_compressed(f"{OUT}/{subj}_geometry.npz", d_left_mm=dL.astype(np.float32), d_both_mm=dB.astype(np.float32), d_right_mm=dR.astype(np.float32), d_left_cone27_mm=dLc.astype(np.float32), d_both_cone27_mm=dBc.astype(np.float32),
                        angle_left_deg=aL.astype(np.float32), angle_both_deg=aB.astype(np.float32), label=lab_v, hemi=hemi_v,
                        frame_xyz=Fv.astype(np.float32), vox_ras=vox.astype(np.float32), nearest_left=jL.astype(np.int32),
                        sensor_left_ras=pl["P"].astype(np.float32), sensor_right_ras=pr["P"].astype(np.float32),
                        sensor_left_frame=pl["F"].astype(np.float32), sensor_right_frame=pr["F"].astype(np.float32), d_any_surface_mm=d_any.astype(np.float32))
    info = {"subject": subj, "uts": S["uts"], "V": int(V), "voxel_mm": S["voxel_mm"].tolist(), "voxel_mm3": float(np.prod(S["voxel_mm"])),
            "frame_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))), "frame_origin_ras": p0.tolist(),
            "params": {k: getattr(G, k) for k in ("HEAD_THR", "HEAD_CLOSE_IT", "PINNA_EXCESS_MM", "FIT_MARGIN_MM", "FIT_INLIER_MM", "WIN_UP_MM", "WIN_BACK_MM", "EAM_R_MM", "EAM_DZ_MM")}}
    for nm, pp in (("left", pl), ("right", pr)):
        info[nm] = {"landmarks": pp["L"], "n_lateral_surface_pts": int(len(pp["P_all"])), "n_box": int(pp["box"].sum()),
                    "n_box_pinna": int((pp["box"] & pp["pin"]).sum()), "n_box_eam": int((pp["box"] & pp["eam"]).sum()),
                    "n_sensor_pts": int(pp["sel"].sum()), "n_pinna_replaced_by_fitted_scalp": pp["n_pinna_replaced"], "scalp_fit_rmse_mm_in_box": pp["fit_rmse_mm"], "sensor_area_cm2_projected": float(pp["sel"].sum() / 100.0),
                    "sensor_frame_x_range": [float(pp["F"][:, 0].min()), float(pp["F"][:, 0].max())],
                    "sensor_frame_y_range": [float(pp["F"][:, 1].min()), float(pp["F"][:, 1].max())],
                    "sensor_frame_z_range": [float(pp["F"][:, 2].min()), float(pp["F"][:, 2].max())]}
    def dstats(d, a, msk):
        q = lambda x, p: float(np.percentile(x, p)) if len(x) else None
        return {"n": int(msk.sum()), "min_mm": float(d[msk].min()), "p5_mm": q(d[msk], 5), "median_mm": q(d[msk], 50),
                "n_le_40mm": int((d[msk] <= 40).sum()), "n_le_60mm": int((d[msk] <= 60).sum()), "n_le_80mm": int((d[msk] <= 80).sum()),
                "frac_angle_gt_27deg_within_80mm": float(np.mean(a[msk & (d <= 80)] > 27)) if (msk & (d <= 80)).any() else None}
    allm = np.ones(V, bool)
    info["dist_left"] = dstats(dL, aL, allm); info["dist_both"] = dstats(dB, aB, allm)
    info["dist_left_lh_only"] = dstats(dL, aL, hemi_v == "lh")
    info["dist_left_cone27"] = {"n_le_80mm": int((dLc <= 80).sum()), "n_le_60mm": int((dLc <= 60).sum()), "median_extra_mm_vs_nearest": float(np.median((dLc - dL)[dLc <= 80]))}
    wm_, pia_, _ = S["surf"]["lh"]; dpia = cKDTree(pl["P"]).query(pia_)[0]
    info["left_pial_vertex_dist_mm"] = {"min": float(dpia.min()), "p1": float(np.percentile(dpia, 1)), "p5": float(np.percentile(dpia, 5)), "median": float(np.median(dpia))}
    info["min_dist_any_surface_mm"] = float(d_any.min())
    info["areas_n"] = {k: int(len(v)) for k, v in sets.items()}
    json.dump(info, open(f"{OUT}/{subj}_sensor.json", "w"), indent=1, default=float)
    figure(subj, S, pl, pr, Fv, dL, lab_v, hemi_v, R, p0)
    print(json.dumps({k: info[k] for k in ("subject", "V", "frame_rotation_deg", "dist_left", "dist_both", "min_dist_any_surface_mm")}), flush=True)
    print("  left sensor", json.dumps(info["left"]["landmarks"]), info["left"]["n_sensor_pts"], "pts; right", info["right"]["n_sensor_pts"])


def figure(subj, S, pl, pr, Fv, dL, lab_v, hemi_v, R, p0):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    L = pl["L"]; fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), gridspec_kw={"width_ratios": [1.25, 1, 0.8]})
    # A: lateral view (head frame y-z), left head surface + sensor + pial coloured by distance
    a = ax[0]; Fa = pl["F_all"]; k = np.arange(0, len(Fa), 7)
    a.scatter(Fa[k, 1], Fa[k, 2], s=0.3, c="0.85", rasterized=True)
    wm, pia, _ = S["surf"]["lh"]; Fp = G.to_frame(pia, R, p0); lat = Fp[:, 0] < np.percentile(Fp[:, 0], 45); kk = np.where(lat)[0][::4]
    tree = cKDTree(pl["P"]); dp = tree.query(pia[kk])[0]
    sc = a.scatter(Fp[kk, 1], Fp[kk, 2], s=0.6, c=np.clip(dp, 10, 90), cmap="viridis_r", vmin=10, vmax=90, rasterized=True)
    Fs = pl["F"]; e = np.arange(0, len(Fs), 3)
    a.scatter(Fs[e, 1], Fs[e, 2], s=0.4, c="#d62728", alpha=0.25, rasterized=True)
    y0, y1, z0, z1 = L["y_EAM"] - G.WIN_BACK_MM, L["y_TP"], L["z_ZA"], L["z_ZA"] + G.WIN_UP_MM
    a.plot([y0, y1, y1, y0, y0], [z0, z0, z1, z1, z0], color="#d62728", lw=1.2)
    t = np.linspace(0, 2 * np.pi, 100); a.plot(L["y_EAM"] + G.EAM_R_MM * np.cos(t), L["z_EAM"] + G.EAM_R_MM * np.sin(t), color="k", lw=0.8, ls="--")
    for nm, (yy, zz) in {"TP": (L["y_TP"], z0 + 5), "HG (ear plane)": (L["y_HG"], L["z_HG"])}.items():
        a.plot(yy, zz, "k+", ms=9); a.text(yy, zz + 2, nm, fontsize=7)
    a.axhline(z0, color="k", lw=0.5, ls=":"); a.text(Fa[:, 1].min(), z0 + 1, "arch level = inferior extent of lateral temporal lobe", fontsize=6)
    a.set_aspect("equal"); a.set_xlabel("head-frame y (mm, anterior →)"); a.set_ylabel("head-frame z (mm, superior)")
    a.set_title(f"{subj}: left lateral view. Red box/points = sensor (pinna, ear-canal disk removed);\nleft pial surface coloured by distance to nearest sensor point (mm)", fontsize=8, loc="left")
    plt.colorbar(sc, ax=a, fraction=0.035, label="distance (mm)")
    # B: axial band through Heschl's gyrus level: voxels coloured by distance, sensor points of both sides
    b = ax[1]; band = np.abs(Fv[:, 2] - L["z_HG"]) < 2.0
    sc2 = b.scatter(Fv[band, 0], Fv[band, 1], s=6, c=np.clip(dL[band], 10, 90), cmap="viridis_r", vmin=10, vmax=90, marker="s")
    for pp in (pl, pr):
        bb = np.abs(pp["F_all"][:, 2] - L["z_HG"]) < 1.0; b.scatter(pp["F_all"][bb, 0], pp["F_all"][bb, 1], s=0.5, c="0.6")
    ss = np.abs(pl["F"][:, 2] - L["z_HG"]) < 1.0; b.scatter(pl["F"][ss, 0], pl["F"][ss, 1], s=2, c="#d62728")
    b.set_aspect("equal"); b.set_xlabel("x (mm, right →)"); b.set_ylabel("y (mm, anterior)")
    b.set_title("axial band at Heschl's gyrus level (±2 mm):\nfunctional voxels coloured by distance to left sensor", fontsize=8, loc="left")
    # C: distance histogram by hemisphere
    c = ax[2]; bins = np.arange(0, 160, 4)
    c.hist(dL[hemi_v == "lh"], bins=bins, color="#2a78d6", alpha=0.8, label="left-hemisphere voxels")
    c.hist(dL[hemi_v == "rh"], bins=bins, color="#eb6834", alpha=0.6, label="right-hemisphere voxels")
    c.axvline(10, color="k", lw=0.6, ls=":"); c.axvline(80, color="k", lw=0.8, ls="--"); c.text(81, c.get_ylim()[1] * 0.9, "8 cm validity limit", fontsize=7)
    c.set_xlabel("distance to nearest left-sensor point (mm)"); c.set_ylabel("voxels"); c.legend(fontsize=7, frameon=False)
    c.set_title("voxel distances (left temple)", fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/{subj}_sanity.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["S1", "S2", "S3"]: main(s)
