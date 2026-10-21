"""Phase C geometry library (CPU): head surface, head frame, brain landmarks, temporal-window sensor patch,
voxel-to-sensor distances. All coordinates in the subject's scanner RAS (mm) unless stated; the "head frame" is the
subject's own mm rotated into MNI305 orientation (x right, y anterior, z superior) without any scaling.

Inputs per subject (OpenNeuro ds003020 derivatives, downloaded to $DATA_DIR/anat/<UTS>/, see README):
  FreeSurfer mri/T1.mgz, mri/transforms/talairach.xfm, surf/{lh,rh}.{pial,white}, label/{lh,rh}.aparc.annot;
  pycortex transforms/<UTS>_auto/{mask_thick.nii.gz, matrices.xfm}.
Conventions copied from Phase B atlas_sets.py (checked there): hf5 column c <-> mask_thick voxel ijk[c] with
ijk = argwhere(mask.T)[:, ::-1]; FreeSurfer surface + c_ras = pycortex surface space = scanner RAS;
pycortex 'coord' maps scanner RAS -> functional voxel index; talairach.xfm maps scanner RAS -> MNI305.
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import json, numpy as np, nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree

DL = f"{VP.DATA}/anat"
UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}

# ------------------------------------------------------------------ fixed parameters (DESIGN_C.md section 2)
HEAD_THR = 15.0          # T1 intensity threshold for the head mask (Phase B atlas_sets.py)
HEAD_CLOSE_IT = 3        # binary closing iterations (Phase B)
PINNA_EXCESS_MM = 6.0    # a lateral surface point standing > 6 mm proud of the fitted scalp surface is ear (pinna), not scalp
FIT_MARGIN_MM = 25.0     # the scalp surface is fitted (robust quadratic) over the window box enlarged by this margin
FIT_INLIER_MM = 3.0      # robust-fit inlier band
WIN_UP_MM = 50.0         # window height above the arch level (coordinator: 4-5 cm; upper end, favours more voxels)
WIN_BACK_MM = 25.0       # window extends this far behind the ear-canal coronal plane ("behind the ear")
EAM_R_MM = 15.0          # ear-canal exclusion disk radius (no sensor on the concha / canal opening)
EAM_DZ_MM = -4.0         # ear-canal centre relative to the arch level (porion ~ arch upper border; canal centre ~4 mm lower)
LAT_TEMPORAL = ["superiortemporal", "middletemporal", "inferiortemporal", "temporalpole"]


def load_subject(subj):
    u = UTS[subj]; d = f"{DL}/{u}"
    T1 = nib.load(f"{d}/mri/T1.mgz")
    lines = open(f"{d}/mri/transforms/talairach.xfm").read().split("Linear_Transform =")[1].strip().rstrip(";").split()
    tal = np.vstack([np.array(lines, float).reshape(3, 4), [0, 0, 0, 1]])                 # scanner RAS -> MNI305
    hdr = T1.header; c_ras = (hdr.get_vox2ras() @ np.linalg.inv(hdr.get_vox2ras_tkr()))[:3, 3]
    surf, annot = {}, {}
    for h in ("lh", "rh"):
        wm = nib.freesurfer.read_geometry(f"{d}/surf/{h}.white")[0].astype(np.float64) + c_ras
        pia, faces = nib.freesurfer.read_geometry(f"{d}/surf/{h}.pial"); pia = pia.astype(np.float64) + c_ras
        lab, ctab, names = nib.freesurfer.read_annot(f"{d}/label/{h}.aparc.annot")
        nm = np.array([n.decode() for n in names]); annot[h] = np.where(lab >= 0, nm[np.clip(lab, 0, None)], "unknown")
        surf[h] = (wm, pia, faces)
    mimg = nib.load(f"{d}/pycortex/mask_thick.nii.gz"); m = np.asarray(mimg.get_fdata()) > 0
    ijk = np.argwhere(m.T)[:, ::-1]                                                        # hf5 column order (Phase A/B)
    xfm = np.array(json.load(open(f"{d}/pycortex/matrices.xfm"))["coord"])               # scanner RAS -> functional voxel
    inv = np.linalg.inv(xfm); vox_ras = ijk @ inv[:3, :3].T + inv[:3, 3]                  # functional voxel centres, scanner RAS
    return dict(subj=subj, uts=u, T1=T1, A=np.asarray(T1.get_fdata(), np.float32), tal=tal, c_ras=c_ras, surf=surf, annot=annot,
                mask=m, ijk=ijk, xfm=xfm, vox_ras=vox_ras, voxel_mm=np.array(mimg.header.get_zooms()[:3], float))


def head_mask(A):
    head = ndimage.binary_fill_holes(ndimage.binary_closing(A > HEAD_THR, iterations=HEAD_CLOSE_IT))
    lab, n = ndimage.label(head)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def head_frame(tal):
    """rotation R (3x3) and origin p0 (scanner RAS) such that f = R (p - p0) is in MNI305 orientation, native mm.
    R = orthogonal polar factor of talairach.xfm's linear part; p0 = native image of the MNI305 origin."""
    M = tal[:3, :3]; U, s, Vt = np.linalg.svd(M); R = U @ Vt
    p0 = np.linalg.solve(M, -tal[:3, 3])
    return R, p0


def to_frame(P, R, p0): return (np.asarray(P) - p0) @ R.T


def lateral_surface(head, vox2ras, side):
    """outermost head voxel along T1 axis 0 for every (axis1, axis2) cell, seen from the given side.
    FreeSurfer conformed T1 is LIA: axis 0 increases toward the subject's LEFT. Returns points (n,3) scanner RAS
    (voxel centre moved 0.5 voxel outward = the voxel's outer face), cell indices (n,2), height (n,) in voxels."""
    n0 = head.shape[0]; ax0 = np.arange(n0)[:, None, None]
    any_ = head.any(0)
    if side == "left": h = np.max(np.where(head, ax0, -1), 0); off = 0.5
    else: h = np.min(np.where(head, ax0, n0 + 1), 0); off = -0.5
    jj, kk = np.where(any_); hh = h[jj, kk].astype(float)
    vox = np.stack([hh + off, jj, kk], 1)
    P = vox @ vox2ras[:3, :3].T + vox2ras[:3, 3]
    return P, np.stack([jj, kk], 1), hh, np.where(any_, h, np.nan)


def surface_normals(head, P, vox2ras):
    """outward unit normals at points P (scanner RAS) from the gradient of the smoothed head mask"""
    sm = ndimage.gaussian_filter(head.astype(np.float32), 2.0); g = np.stack(np.gradient(sm), -1)     # voxel-space gradient (inward)
    inv = np.linalg.inv(vox2ras); v = np.rint(P @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    v = np.clip(v, 0, np.array(head.shape) - 1)
    # step one voxel inward if the rounded point fell outside the mask's smooth band
    gv = g[v[:, 0], v[:, 1], v[:, 2]]
    n_ras = -(gv @ vox2ras[:3, :3].T)                                          # outward = minus inward gradient, to RAS
    return n_ras / np.maximum(np.linalg.norm(n_ras, axis=1, keepdims=True), 1e-9)


def landmarks(S, hemi, R, p0):
    """brain landmarks in head-frame coordinates (native mm) from the hemisphere's pial surface + Desikan labels"""
    wm, pia, _ = S["surf"][hemi]; lab = S["annot"][hemi]; F = to_frame(pia, R, p0)
    tp = F[lab == "temporalpole"]; hg = F[lab == "transversetemporal"]; lt = F[np.isin(lab, LAT_TEMPORAL)]
    out = {"y_TP": float(np.percentile(tp[:, 1], 99.5)),                     # temporal-pole tip (anterior-most, robust)
           "y_HG": float(np.median(hg[:, 1])),                                # Heschl's gyrus -> ear-canal coronal level
           "z_HG": float(np.median(hg[:, 2])),
           "z_inf_temporal": float(np.percentile(lt[:, 2], 2.0)),            # inferior extent of the lateral temporal lobe
           "x_lat_temporal": float(np.percentile(lt[:, 0], 2.0 if hemi == "lh" else 98.0)),
           "n_tp": int(len(tp)), "n_hg": int(len(hg)), "n_lt": int(len(lt))}
    out["y_EAM"] = out["y_HG"]; out["z_ZA"] = out["z_inf_temporal"]; out["z_EAM"] = out["z_ZA"] + EAM_DZ_MM
    return out


def window_patch(P, F, cells, height, side, L):
    """lateral-surface points inside the window box (head frame) on the given side"""
    y, z = F[:, 1], F[:, 2]
    box = (y <= L["y_TP"]) & (y >= L["y_EAM"] - WIN_BACK_MM) & (z >= L["z_ZA"]) & (z <= L["z_ZA"] + WIN_UP_MM)
    lat = (F[:, 0] < 0) if side == "left" else (F[:, 0] > 0)
    return box & lat


def scalp_fit(F, side, L):
    """robust quadratic fit x = h(y, z) of the lateral head surface around the window (head frame). Returns
    (coef, proud) where proud = how far each point stands out laterally beyond the fit (mm, > 0 = more lateral)."""
    y, z, x = F[:, 1], F[:, 2], F[:, 0]
    reg = (y <= L["y_TP"] + FIT_MARGIN_MM) & (y >= L["y_EAM"] - WIN_BACK_MM - FIT_MARGIN_MM) & (z >= L["z_ZA"] - FIT_MARGIN_MM) & (z <= L["z_ZA"] + WIN_UP_MM + FIT_MARGIN_MM)
    reg &= (x < 0) if side == "left" else (x > 0)
    D = lambda yy, zz: np.stack([np.ones_like(yy), yy, zz, yy * yy, yy * zz, zz * zz], 1)
    use = reg.copy()
    for _ in range(8):
        coef = np.linalg.lstsq(D(y[use], z[use]), x[use], rcond=None)[0]
        proud = (D(y, z) @ coef - x) if side == "left" else (x - D(y, z) @ coef)
        new = reg & (np.abs(proud) <= FIT_INLIER_MM)
        if (new == use).all(): break
        use = new
    return coef, proud, D


def fit_normals(F, coef, side):
    """outward unit normals (head frame) of the fitted scalp x = h(y,z) at the points' (y, z)"""
    y, z = F[:, 1], F[:, 2]; hy = coef[1] + 2 * coef[3] * y + coef[4] * z; hz = coef[2] + coef[4] * y + 2 * coef[5] * z
    n = np.stack([-np.ones_like(y), hy, hz], 1) if side == "left" else np.stack([np.ones_like(y), -hy, -hz], 1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)
