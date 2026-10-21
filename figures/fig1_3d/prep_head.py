"""ISBI 2027 Figure 1A (2026-10-20): meshes for a 3D cut-away of participant S1's head at the left temporal window.
Layers follow the link budget's assumptions, not a skull segmentation: scalp and muscle = the outer 5 mm of the T1 head
mask, temporal bone = the next 2.9 mm, brain = tissue inside the pial surfaces (FreeSurfer), coloured by T1 intensity so
the cut face shows grey and white matter. The probe sits on the real Phase C window patch; the target is a real voxel of
the anatomical preset about 5 cm from it, chosen to lie straight along the probe's inward direction.
A corner of skin, bone and brain above an (almost axial) plane through the beam axis is removed, so the beam, the layers
and the voxel all lie on the exposed shelf.
Output: fig1_3d/head_S1.npz (meshes in the head frame, mm: x right, y anterior, z superior) + anchors.json.
Inputs: $WORK_DIR/geometry/S1_geometry.npz, S1_sensor.json (linkbudget/build_geometry.py), $WORK_DIR/inputs/S1/anat_cols.npy,
$DATA_DIR/anat/UTS01 (linkbudget/geom_lib.py). head_S1.npz and context_S1.npz (participant-MRI-derived meshes) are not in the repository.
Run: python prep_head.py   (nibabel, scipy, scikit-image 0.26, Pillow)"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, numpy as np, nibabel as nib
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import measure
GEO = f"{VP.WORK}/geometry"
E2 = VP.INPUTS
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(VP.ROOT, "linkbudget"))
import geom_lib as G

SCALP_MM, BONE_MM = 5.0, 2.9          # link budget at 5 cm (Fig. 1 budget rows)
TARGET_MM = 50.0

S = G.load_subject("S1"); A = S["A"]; v2r = S["T1"].header.get_vox2ras(); R, p0 = G.head_frame(S["tal"])
head = G.head_mask(A)
g = np.load(f"{GEO}/S1_geometry.npz"); sens = g["sensor_left_ras"].astype(float); vox = g["vox_ras"].astype(float)
d = g["d_left_mm"]; jn = g["nearest_left"]; ang = g["angle_left_deg"]
anat = np.load(f"{E2}/S1/anat_cols.npy")

# ---- probe point and target voxel: preset voxel 48-52 mm from the window whose path is closest to the inward normal
# (smallest incidence angle); the probe sits on that voxel's nearest window point
cen = sens.mean(0); cen_pt = sens[np.argmin(np.linalg.norm(sens - cen, axis=1))]
ok = np.zeros(len(d), bool); ok[anat] = True
ok &= (d >= TARGET_MM - 2) & (d <= TARGET_MM + 2)
cand = np.where(ok)[0]; best = cand[np.argmin(ang[cand])]
probe_ras = sens[jn[best]]; vox_ras = vox[best]
print("voxel", best, "label", g["label"][best], "d_mm %.1f" % d[best], "angle %.1f" % ang[best])

F = lambda P: G.to_frame(P, R, p0)
pf, vf = F(probe_ras[None])[0], F(vox_ras[None])[0]
b = (vf - pf) / np.linalg.norm(vf - pf)
yv = np.array([0, 1.0, 0]); n = np.cross(b, yv); n /= np.linalg.norm(n)
if n[2] < 0: n = -n                                              # cut-plane normal points up

# ---- voxel grid in the head frame for the masks: T1 voxel centres -> frame
ijk = np.indices(A.shape).reshape(3, -1).T
def vox2frame(ijk_): return F(ijk_ @ v2r[:3, :3].T + v2r[:3, 3])
dist = ndimage.distance_transform_edt(head)                      # mm (1-mm voxels)

# brain solid: rasterise both pial surfaces, close, fill
brain = np.zeros(A.shape, bool); inv = np.linalg.inv(v2r)
for h in ("lh", "rh"):
    wm, pia, faces = S["surf"][h]
    pts = np.concatenate([pia, (pia[faces[:, 0]] + pia[faces[:, 1]] + pia[faces[:, 2]]) / 3])
    q = np.rint(pts @ inv[:3, :3].T + inv[:3, 3]).astype(int)
    shell = np.zeros(A.shape, bool); shell[tuple(q.T)] = True
    shell = ndimage.binary_dilation(shell, iterations=1)
    brain |= ndimage.binary_fill_holes(shell)
# keep tissue, drop sulcal CSF (FreeSurfer-normalised T1: CSF < 40, grey matter 60-90, white matter ~110)
brain &= ndimage.gaussian_filter(A, 0.6) > 52
lab, nl = ndimage.label(brain); sz = np.bincount(lab.ravel()); sz[0] = 0
brain = np.isin(lab, np.argsort(sz)[-2:])                          # the two hemispheres
brain = ndimage.binary_fill_holes(brain)                            # ventricles
brain &= dist > SCALP_MM + BONE_MM + 0.5
scalp = head & (dist <= SCALP_MM)
bone = head & (dist > SCALP_MM) & (dist <= SCALP_MM + BONE_MM)

# ---- cut: a slot along the beam, open on top: above the plane through the beam axis, within +-27 mm across the beam,
# from outside the scalp to 14 mm past the voxel
fr = vox2frame(ijk).reshape(A.shape + (3,))
side = np.cross(n, b); side /= np.linalg.norm(side); L = float(np.linalg.norm(vf - pf))
rel = fr - pf
cut = ((rel @ n) > 0) & (np.abs(rel @ side) < 27.0) & ((rel @ b) < L + 14.0) & ((rel @ b) > -40.0)
for m in (scalp, bone, brain): m &= ~cut

def mesh(mask, sigma=1.3, step=1):
    sl = ndimage.find_objects(mask.astype(np.int8))[0]
    lo = np.array([s.start for s in sl]) - 3; hi = np.array([s.stop for s in sl]) + 3
    lo = np.maximum(lo, 0); hi = np.minimum(hi, A.shape)
    sub = ndimage.gaussian_filter(mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32), sigma)
    vv, ff, nn, _ = measure.marching_cubes(sub, 0.5, step_size=step)
    vv = vv + lo
    return F(vv @ v2r[:3, :3].T + v2r[:3, 3]), ff.astype(np.int32), vv

out = {}
for name, m in (("scalp", scalp), ("bone", bone), ("brain", brain)):
    V, Fa, vv = mesh(m, 0.9 if name == "brain" else 1.3)
    out[f"{name}_v"] = V.astype(np.float32); out[f"{name}_f"] = Fa
    print(name, len(V), "verts", len(Fa), "faces")
    if name == "scalp":                                          # temporal-window patch on the skin
        dw, _ = cKDTree(F(sens)).query(V); out["scalp_win"] = (dw < 3.0).astype(np.float32)
    if name == "brain":                                          # T1 intensity a little inside the surface
        T = ndimage.gaussian_filter(A, 0.7)
        cm = ndimage.center_of_mass(m)
        gx = np.gradient(ndimage.gaussian_filter(m.astype(np.float32), 1.5))
        gi = np.stack([ndimage.map_coordinates(gc, vv.T, order=1) for gc in gx], 1)
        gi /= np.maximum(np.linalg.norm(gi, axis=1, keepdims=True), 1e-6)
        inside = vv + 1.2 * gi                                    # gradient of the mask points inward
        out["brain_t1"] = ndimage.map_coordinates(T, inside.T, order=1).astype(np.float32)

# window outline on the scalp (frame coords): boundary of the patch in its (y, z) footprint
sf = F(sens); yz = sf[:, 1:]
from scipy.spatial import ConvexHull
hull = ConvexHull(yz); loop = sf[hull.vertices]
out["window_loop"] = loop.astype(np.float32)
out["probe"] = pf.astype(np.float32); out["voxel"] = vf.astype(np.float32); out["beam_dir"] = b.astype(np.float32)
out["cut_n"] = n.astype(np.float32)
np.savez_compressed(f"{OUT}/head_S1.npz", **out)
json.dump({"voxel_index": int(best), "label": str(g["label"][best]), "d_left_mm": float(d[best]), "angle_deg": float(ang[best]),
           "probe_frame": pf.tolist(), "voxel_frame": vf.tolist(), "beam_dir": b.tolist(), "cut_normal": n.tolist(),
           "scalp_mm": SCALP_MM, "bone_mm": BONE_MM, "window_area_cm2": json.load(open(f"{GEO}/S1_sensor.json"))["left"]["sensor_area_cm2_projected"]},
          open(f"{OUT}/anchors.json", "w"), indent=1)
print("done")

# ---- context view (added 2026-10-20, second design): back half of the head shell + the T1 slice on a coronal-ish plane
# through the beam axis (plane spanned by the beam and the in-plane vertical), as an RGBA texture
e1 = b; zz = np.array([0, 0, 1.0]); e2 = zz - (zz @ e1) * e1; e2 /= np.linalg.norm(e2); nn = np.cross(e1, e2)
if nn[1] < 0: nn = -nn                                            # plane normal points anterior (towards the camera)
Vh, Fh, _ = mesh(head, 1.5)
keep = ((Vh[Fh] - pf) @ nn).max(1) < 0.0                           # faces behind the plane
out2 = {"head_v": Vh.astype(np.float32), "head_f": Fh[keep].astype(np.int32)}
U0, U1, V0, V1, step = -14.0, 168.0, -62.0, 98.0, 0.3                 # plane coords from the probe point (mm)
gu = np.arange(U0, U1 + 1e-6, step); gv = np.arange(V1, V0 - 1e-6, -step)
UU, VV = np.meshgrid(gu, gv)
P = pf + UU[..., None] * e1 + VV[..., None] * e2
rasP = P @ R + p0; ijkP = (rasP.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]).T
T1s = ndimage.map_coordinates(A, ijkP, order=1).reshape(UU.shape)
hm = ndimage.map_coordinates(ndimage.gaussian_filter(head.astype(np.float32), 1.0), ijkP, order=1).reshape(UU.shape)
g8 = np.clip(T1s / 125.0, 0, 1) ** 0.85
rgba = np.dstack([g8, g8, g8, np.clip((hm - 0.35) / 0.3, 0, 1)])
from PIL import Image
Image.fromarray((rgba * 255).astype(np.uint8), "RGBA").save(f"{OUT}/slice_S1.png")
corners = np.array([pf + U0 * e1 + V1 * e2, pf + U1 * e1 + V1 * e2, pf + U1 * e1 + V0 * e2, pf + U0 * e1 + V0 * e2])
out2.update(slice_corners=corners.astype(np.float32), e1=e1.astype(np.float32), e2=e2.astype(np.float32), nn=nn.astype(np.float32),
            probe=pf.astype(np.float32), voxel=vf.astype(np.float32), beam_dir=b.astype(np.float32))
np.savez_compressed(f"{OUT}/context_S1.npz", **out2)
print("context done", len(Fh), "->", int(keep.sum()), "faces")
