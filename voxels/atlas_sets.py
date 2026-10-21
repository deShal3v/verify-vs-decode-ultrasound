"""Phase B area sets (PLAN 1.7; DESIGN.md section 7): a-priori voxel sets in each subject's hf5 column order.
Mapping: FreeSurfer vertex labels -> functional voxels by sampling 6 points from white (pycortex wm) to pial per vertex,
through pycortex's 'coord' transform (surface coordinates -> functional voxel indices), majority vote per voxel.
fsaverage atlases (HCP-MMP1, LanA) -> subject vertices by nearest neighbour on sphere.reg (nilearn fsaverage sphere).
Temporal window: T1 head mask, ray from the head centroid to MNI305 (+-70,-15,-5) (talairach.xfm), scalp entry point,
inward normal of the smoothed mask, 20 mm diameter cylinder, depth <= 60 mm, both sides.
Output $WORK_DIR/<S>/areas.json: {set: [hf5 columns]}, controls (3 size-matched random subsets per set), checks."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, numpy as np, nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
SUBJ = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
SPEECH_HCP = ["6v", "6r", "55b", "44", "45"]
SEM = ["superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus", "superiortemporal", "middletemporal", "inferiortemporal",
       "bankssts", "fusiform", "transversetemporal", "entorhinal", "temporalpole", "parahippocampal", "lateraloccipital", "lingual", "cuneus",
       "pericalcarine", "posteriorcingulate", "isthmuscingulate"]
PFC = ["superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis", "parstriangularis", "parsorbitalis", "lateralorbitofrontal",
       "medialorbitofrontal", "precentral", "paracentral", "frontalpole", "rostralanteriorcingulate", "caudalanteriorcingulate"]
AUD = ["transversetemporal", "superiortemporal"]
LANA_THR = 0.20; WIN_MNI = [(-70.0, -15.0, -5.0), (70.0, -15.0, -5.0)]; WIN_R = 10.0; WIN_DEPTH = 60.0

def main(subj):
    uts = SUBJ[subj]; pc = f"{VP.DS}/pycortex-db/{uts}"; fs = f"{VP.DS}/freesurfer_subjdir/{uts}"
    out_dir = f"{VP.WORK}/{subj}"; chk = {}
    m = np.asarray(nib.load(f"{pc}/transforms/{uts}_auto/mask_thick.nii.gz").get_fdata()) > 0
    ijk = np.argwhere(m.T)[:, ::-1]; V = len(ijk)                       # hf5 column c <-> nifti voxel ijk[c] (Phase A order)
    lut = -np.ones(m.shape, np.int64); lut[tuple(ijk.T)] = np.arange(V)
    xfm = np.array(json.load(open(f"{pc}/transforms/{uts}_auto/matrices.xfm"))["coord"])
    def to_vox(P):
        q = P @ xfm[:3, :3].T + xfm[:3, 3]; return q
    labels_vox = {}
    votes = {}   # (atlas) -> list of (voxel col, label)
    hemis = ["lh", "rh"]
    surf = {}
    for h in hemis:
        # FreeSurfer white/pial + c_ras = pycortex's surface space (scanner RAS; checked on UTS01: pycortex pia - lh.pial = c_ras exactly)
        T1h = nib.load(f"{fs}/mri/T1.mgz").header; c_ras = (T1h.get_vox2ras() @ np.linalg.inv(T1h.get_vox2ras_tkr()))[:3, 3]
        wm = nib.freesurfer.read_geometry(f"{fs}/surf/{h}.white")[0].astype(np.float64) + c_ras
        pia = nib.freesurfer.read_geometry(f"{fs}/surf/{h}.pial")[0].astype(np.float64) + c_ras
        chk[f"{h}_n_vertices"] = [len(wm), len(pia)]; chk["c_ras"] = c_ras.tolist()
        surf[h] = (wm, pia)
    # axis-order check: fraction of samples inside the mask for (x,y,z) vs reversed
    for order in ("xyz", "zyx"):
        inside = []
        for h in hemis:
            wm, pia = surf[h]; q = np.rint(to_vox(0.5 * (wm + pia))).astype(int)
            if order == "zyx": q = q[:, ::-1]
            ok = (q >= 0).all(1) & (q < np.array(m.shape)).all(1); inside.append(m[tuple(q[ok].T)].mean() * ok.mean())
        chk[f"midthickness_in_mask_{order}"] = float(np.mean(inside))
    assert chk["midthickness_in_mask_xyz"] > chk["midthickness_in_mask_zyx"], chk
    vert_vox = {}
    for h in hemis:
        wm, pia = surf[h]; cols = []
        for t in np.linspace(0, 1, 6):
            q = np.rint(to_vox(wm + t * (pia - wm))).astype(int)
            ok = (q >= 0).all(1) & (q < np.array(m.shape)).all(1)
            c = -np.ones(len(q), np.int64); c[ok] = lut[tuple(q[ok].T)]; cols.append(c)
        vert_vox[h] = np.stack(cols, 1)                                   # (nvert, 6) hf5 column or -1
    def vox_from_vertex_labels(lab_by_hemi, names):
        """lab_by_hemi: {h: (nvert,) label strings or ints}; names: set of wanted labels -> sorted hf5 columns (majority vote)"""
        cnt_in = np.zeros(V); cnt_all = np.zeros(V)
        for h in hemis:
            want = np.isin(lab_by_hemi[h], list(names))
            vv = vert_vox[h]
            for j in range(vv.shape[1]):
                ok = vv[:, j] >= 0
                np.add.at(cnt_all, vv[ok, j], 1); np.add.at(cnt_in, vv[ok & want, j], 1)
        return np.where((cnt_all > 0) & (cnt_in / np.maximum(cnt_all, 1) > 0.5))[0]
    # native Desikan
    aparc = {}
    for h in hemis:
        lab, ctab, names = nib.freesurfer.read_annot(f"{fs}/label/{h}.aparc.annot")
        nm = np.array([n.decode() for n in names]); aparc[h] = np.where(lab >= 0, nm[np.clip(lab, 0, None)], "unknown")
    # fsaverage -> subject (nearest neighbour on sphere.reg)
    from nilearn import datasets
    fsa = datasets.fetch_surf_fsaverage("fsaverage", data_dir=f"{VP.ATLAS}/nilearn")
    hcp, lana = {}, {}
    for h, side in (("lh", "left"), ("rh", "right")):
        sph_s, _ = nib.freesurfer.read_geometry(f"{fs}/surf/{h}.sphere.reg")
        sph_a = nib.load(str(fsa[f"sphere_{side}"])).darrays[0].data.astype(np.float64)
        nn = cKDTree(sph_a / np.linalg.norm(sph_a, axis=1, keepdims=True)).query(sph_s / np.linalg.norm(sph_s, axis=1, keepdims=True))[1]
        lab, ctab, names = nib.freesurfer.read_annot(f"{VP.ATLAS}/{h}.HCP-MMP1.annot")
        nm = np.array([n.decode() for n in names]); full = np.where(lab >= 0, nm[np.clip(lab, 0, None)], "???")
        hcp[h] = np.array([x.split("_")[1] if x.count("_") >= 2 else x for x in full[nn]])     # 'L_6v_ROI' -> '6v'
        p = np.asarray(nib.load(f"{VP.ATLAS}/lana/FS/{'LH' if h == 'lh' else 'RH'}_LanA_n804.nii.gz").get_fdata()).ravel()
        chk[f"{h}_fsaverage_nvert"] = [int(len(sph_a)), int(len(p)), int(len(lab))]
        lana[h] = np.where(p[nn] >= LANA_THR, "lana", "no")
    tang = json.load(open(f"{VP.TANG_BOX}/data_train/ROIs/{subj}.json"))
    sets = {}
    speech_hcp = vox_from_vertex_labels(hcp, SPEECH_HCP)
    sets["speech_motor"] = np.union1d(speech_hcp, np.array(tang["speech"]))
    sets["auditory"] = np.union1d(vox_from_vertex_labels(aparc, AUD), np.array(tang["auditory"]))
    sets["language_LanA"] = vox_from_vertex_labels(lana, ["lana"])
    sets["semantic"] = np.setdiff1d(vox_from_vertex_labels(aparc, SEM), sets["speech_motor"])
    sets["prefrontal"] = np.setdiff1d(vox_from_vertex_labels(aparc, PFC), sets["speech_motor"])
    sets["whole_cortex"] = np.arange(V)
    # ---- temporal window (T1 geometry)
    T1 = nib.load(f"{fs}/mri/T1.mgz"); A = np.asarray(T1.get_fdata())
    vox2ras = T1.header.get_vox2ras(); vox2tkr = T1.header.get_vox2ras_tkr()
    tal = np.loadtxt(f"{fs}/mri/transforms/talairach.xfm", skiprows=5, max_rows=3, delimiter=None, comments=";", usecols=(0, 1, 2, 3)) if False else None
    lines = open(f"{fs}/mri/transforms/talairach.xfm").read().split("Linear_Transform =")[1].strip().rstrip(";").split()
    tal = np.vstack([np.array(lines, float).reshape(3, 4), [0, 0, 0, 1]])           # scanner RAS -> MNI305
    head = ndimage.binary_fill_holes(ndimage.binary_closing(A > 15, iterations=3))
    lab_, nlab = ndimage.label(head); head = lab_ == (np.argmax(np.bincount(lab_.ravel())[1:]) + 1)
    sm = ndimage.gaussian_filter(head.astype(np.float32), 2.0); grad = np.stack(np.gradient(sm), -1)
    cen = np.argwhere(head).mean(0)
    # functional voxel centres -> T1 voxel coordinates (mm, conformed 1 mm)
    # pycortex surfaces are FreeSurfer surfaces + c_ras (scanner RAS; checked: pia - lh.pial = c_ras exactly), so the
    # inverse coord transform gives scanner RAS
    inv_xfm = np.linalg.inv(xfm); sras = ijk @ inv_xfm[:3, :3].T + inv_xfm[:3, 3]
    t1v = sras @ np.linalg.inv(vox2ras)[:3, :3].T + np.linalg.inv(vox2ras)[:3, 3]

    win = np.zeros(V, bool); chk["window"] = {}
    for P in WIN_MNI:
        ras = np.linalg.inv(tal) @ np.array([*P, 1.0]); pv = (np.linalg.inv(vox2ras) @ ras)[:3]
        d = (pv - cen) / np.linalg.norm(pv - cen); E = None
        for s in np.arange(0, 200, 0.5):
            x = cen + s * d; xi = np.rint(x).astype(int)
            if (xi < 0).any() or (xi >= np.array(A.shape)).any() or not head[tuple(xi)]:
                break
            E = x
        n_in = grad[tuple(np.rint(E).astype(int))]; n_in = n_in / np.linalg.norm(n_in)
        v = t1v - E; depth = v @ n_in; radial = np.linalg.norm(v - depth[:, None] * n_in[None], axis=1)
        sel = (depth >= 0) & (depth <= WIN_DEPTH) & (radial <= WIN_R); win |= sel
        E_ras = vox2ras @ np.array([*E, 1.0])
        chk["window"][str(P)] = {"entry_T1_vox": E.tolist(), "entry_MNI": (tal @ E_ras)[:3].tolist(), "inward_normal_T1": n_in.tolist(),
                                 "n_vox": int(sel.sum()), "min_depth_mm_of_selected": float(depth[sel].min()) if sel.any() else None}
    sets["temporal_window"] = np.where(win)[0]
    rng = np.random.default_rng({"S1": 11, "S2": 22, "S3": 33}[subj])
    controls = {nm: [np.sort(rng.choice(V, len(v), replace=False)).tolist() for _ in range(3)] for nm, v in sets.items() if nm != "whole_cortex"}
    chk["sizes"] = {k: int(len(v)) for k, v in sets.items()}; chk["tang_speech_in_speech_motor"] = float(np.isin(tang["speech"], sets["speech_motor"]).mean())
    chk["tang_auditory_vs_aparc_overlap"] = float(np.isin(tang["auditory"], vox_from_vertex_labels(aparc, AUD)).mean())
    chk["speech_hcp_only"] = int(len(speech_hcp))
    json.dump({"sets": {k: v.tolist() for k, v in sets.items()}, "controls": controls, "checks": chk,
               "definitions": {"speech_hcp": SPEECH_HCP, "semantic": SEM, "prefrontal": PFC, "auditory": AUD, "lana_thr": LANA_THR,
                               "window_mni": WIN_MNI, "window_radius_mm": WIN_R, "window_depth_mm": WIN_DEPTH}}, open(f"{out_dir}/areas.json", "w"))
    print(json.dumps(chk, indent=1))

if __name__ == "__main__":
    main(sys.argv[1])
