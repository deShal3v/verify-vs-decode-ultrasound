"""Phase C shared helpers (DESIGN_C.md sections 4-6): ultrasound noise conditions, cell-name parsing, rungs.

Ultrasound condition c (usL10, usL50, usL90, usB50, usB90, usL50lo, usL50hi, usL50raw, usL50cone, usL90cone):
  x_v + sqrt(f_v * clip(var_n_v, 0, 1)) * eps_v  on the full voxel grid, eps = Phase B verify.Data.corr_noise (unit
  variance, spatially correlated, this subject's noise_fit.json), seed seed_of(subject, "us", block) for EVERY ultrasound
  condition (common random numbers); then the block is re-z-scored and restricted to the condition's voxel set.
  f_v, the set and var_n come from $WORK_DIR/<S>/us_maps.npz (built on CPU by geometry/snr_maps.py).
Plain conditions (native, m6c, m12c, m18c, m24c, m30c): Phase B's tang_fit.Noise exactly (same seeds), full grid.
Rungs: r1/r4/r15 = first 1/4/15 sessions (Phase B); n1/n2/n3 = first 1/2/3 stories of Tang's order (session 2)."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import verify as Vf
from verify import Data, seed_of

SESS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
PLAIN_DB = {"native": 0.0, "m3c": 3.0, "m6c": 6.0, "m12c": 12.0, "m18c": 18.0, "m24c": 24.0, "m30c": 30.0}
US_PREFIX = "us"


def is_us(cond): return cond.startswith(US_PREFIX)


def parse_cell(cell):
    """'usL50_r15' -> ('usL50', 'r15'); 'native_n3' -> ('native', 'n3')"""
    cond, rung = cell.rsplit("_", 1); assert rung[0] in "rn" and rung[1:].isdigit(), cell
    return cond, rung


def rung_stories(rung, s2s):
    """rung 'r<k>': all stories of the first k sessions (Tang's order); 'n<k>': the first k stories of that order"""
    order = [st for s in SESS for st in s2s[str(s)]]
    k = int(rung[1:])
    if rung[0] == "r": return [st for s in SESS[:k] for st in s2s[str(s)]]
    return order[:k]


def pseudo_db(cond):
    """number used only for ordering/labels (Phase B 'dB' field); ultrasound conditions get None"""
    return None if is_us(cond) else -PLAIN_DB[cond]


class USMaps:
    def __init__(self, subj, work=f"{VP.WORK}"):
        self.subj = subj; p = f"{work}/{subj}/us_maps.npz"; z = np.load(p)
        self.var_n = z["var_n"].astype(np.float64)
        self.conds = sorted({k.split("|")[0] for k in z.files if "|" in k})
        self.cols = {c: z[f"{c}|cols"].astype(np.int64) for c in self.conds}
        self.f = {c: z[f"{c}|f"].astype(np.float64) for c in self.conds}
        node_vn = f"{work}/{subj}/var_noise.npy"
        if os.path.exists(node_vn):                                  # the CPU maps must match this node's prep_resp output
            vn = np.load(node_vn).astype(np.float64); assert vn.shape == self.var_n.shape, (vn.shape, self.var_n.shape)
            dmax = float(np.abs(vn - self.var_n).max()); assert dmax < 1e-3, f"var_noise mismatch {dmax}"

    def sig_full(self, cond, V):
        """per-voxel added-noise SD on the full grid (0 outside the set)"""
        s = np.zeros(V); c = self.cols[cond]
        s[c] = np.sqrt(self.f[cond] * np.clip(self.var_n[c], 0, 1)); return s


class USNoise:
    """noise for one subject: apply(x, cond, key, restrict) -> torch float32 (n, V_set or V)"""
    def __init__(self, subj, work=f"{VP.WORK}", dev=None):
        self.dev = dev or Vf.DEV; d = f"{work}/{subj}"; self.subj = subj
        D = Data.__new__(Data); D.subj = subj
        info = json.load(open(f"{d}/index.json")); D.V = info["V"]
        D.nf = json.load(open(f"{d}/noise_fit.json")); D.ijk = torch.tensor(np.load(f"{d}/vox_ijk.npy").astype(np.int64), device=self.dev)
        self.D = D; self.V = D.V; self.var_n = np.load(f"{d}/var_noise.npy")
        self.maps = USMaps(subj, work) if os.path.exists(f"{d}/us_maps.npz") else None
        self._sig = {}

    def cols(self, cond):
        return None if not is_us(cond) else self.maps.cols[cond]

    def sig(self, cond):
        if cond not in self._sig:
            if is_us(cond): s = self.maps.sig_full(cond, self.V)
            else: s = np.sqrt((10 ** (PLAIN_DB[cond] / 10) - 1) * np.clip(self.var_n, 0, 1))
            self._sig[cond] = torch.from_numpy(s.astype(np.float32)).to(self.dev)
        return self._sig[cond]

    def apply(self, x, cond, key, restrict=True):
        """x: (n, V) numpy/torch z-scored block. Plain conditions: identical to Phase B tang_fit.Noise.apply / verify.Data.load.
        Ultrasound: noise on the full grid, re-z-score, then (restrict=True) keep the condition's columns only;
        (restrict=False) return the full grid with columns outside the set set to 0 (nothing outside the set can leak)."""
        xt = (torch.from_numpy(np.ascontiguousarray(x)) if isinstance(x, np.ndarray) else x).to(self.dev).float()
        if cond == "native" or (not is_us(cond) and PLAIN_DB[cond] <= 0): return xt
        g = torch.Generator(device=self.dev)
        g.manual_seed(seed_of(self.subj, "us", key) if is_us(cond) else seed_of(self.subj, cond, key))
        eps = self.D.corr_noise(xt.shape[0], g) if (is_us(cond) or cond.endswith("c")) else torch.randn(xt.shape, generator=g, device=self.dev)
        xt = xt + self.sig(cond)[None] * eps; del eps
        xt = (xt - xt.mean(0)) / xt.std(0).clamp_min(1e-6)
        if is_us(cond):
            c = torch.from_numpy(self.maps.cols[cond]).to(self.dev)
            if restrict: return xt[:, c].contiguous()
            out = torch.zeros_like(xt); out[:, c] = xt[:, c]; return out
        return xt
