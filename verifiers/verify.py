"""Track L Phase A: linear contrastive forward verifier, fMRI -> GTR-T5 embedding of the heard 6 s window.

Brain input for anchor a (window ending at T_a = 2a+12 s): PCA projections of response rows a-1..a+4
(every row in which a window word can appear under Tang's 1-4 TR delays), zero-padded outside the story.
PCA (randomised, all cortical voxels) is fitted on the fold's training rows only.
Model: z = normalise(W x + b), InfoNCE (symmetric, in-batch negatives; same-story pairs with overlapping
text, |da|<=3, are masked out), learnable logit scale, L2 penalty lam*||W||^2, Adam, early stopping on the
calibration story (fromboyhoodtofatherhood; never trained on, never evaluated).
Hyper-parameters (q in QS, lam in LAMS) tuned per (subject, SNR condition, rung) on the calibration story.
Evaluation windows: leave-one-story-out over the 82 training stories (story s scored by a model trained on
rung-stories minus s), plus the held-out test story (10 single-trial repeats; repeat 0 = Tang's ds004510 trial).
Readouts: 6 s windows and 20 s segments (mean of 10 sub-anchor scores). Threshold (yes/no) and softmax
temperature (bits) are fitted per model on the calibration story.
Noise conditions: white Gaussian noise per voxel, variance set from the test-story ncsnr so that per-voxel
SNR drops by 6 or 12 dB; added to every block (train, calibration, test repeats) with fixed seeds, then each
block is re-z-scored; everything (PCA, model, tuning, T, threshold) is refit per condition.
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, math, argparse, hashlib, numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
DEV = "cuda"
SESS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
OFFS = [-1, 0, 1, 2, 3, 4]
QS = [256, 1024]
LAMS = [1e-6, 1e-5, 1e-4, 1e-3]
GRID = [("native", 15), ("native", 4), ("native", 1), ("m6", 15), ("m6", 4), ("m12", 15), ("m12", 4),
        ("m6c", 15), ("m6c", 4), ("m12c", 15), ("m12c", 4),
        ("m6c", 1), ("m3c", 15), ("m18c", 15), ("m18c", 4), ("m24c", 15), ("m24c", 4), ("m12c", 1)]       # Phase B additions
DB = {"native": 0.0, "m6": 6.0, "m12": 12.0, "m6c": 6.0, "m12c": 12.0, "m3c": 3.0, "m18c": 18.0, "m24c": 24.0}   # suffix c = spatially correlated noise model
DBINS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.2]
TXT = f"{VP.WORK}/text"

def seed_of(*xs): return int(hashlib.md5("|".join(map(str, xs)).encode()).hexdigest()[:8], 16)
def sync(): torch.cuda.synchronize()

class Data:
    def __init__(self, subj):
        self.subj = subj; d = f"{VP.WORK}/{subj}"
        self.info = json.load(open(f"{d}/index.json")); self.idx = self.info["index"]
        self.mm = np.load(f"{d}/resp_f16.npy", mmap_mode="r"); self.N, self.V = self.mm.shape
        self.var_n = np.load(f"{d}/var_noise.npy")
        if os.path.exists(f"{d}/noise_fit.json"):
            self.nf = json.load(open(f"{d}/noise_fit.json")); self.ijk = torch.tensor(np.load(f"{d}/vox_ijk.npy").astype(np.int64), device=DEV)
        self.R = torch.empty((self.N, self.V), dtype=torch.float16, device=DEV)
        self.blocks = {}
        for e in self.idx:
            key = e["story"] if e["kind"] in ("train", "calib") else e["kind"]
            self.blocks[key] = (e["start"], e["n"])
        self.train_stories = [e["story"] for e in self.idx if e["kind"] == "train"]
        self.sess = {e["story"]: e["session"] for e in self.idx if e["kind"] == "train"}
    def corr_noise(self, n, g):
        """(n, V) unit-variance noise: independent + global + short- and long-range Gaussian random fields,
        shares and lengths from noise_fit.json (fitted to this subject's residual correlation vs distance)"""
        nf = self.nf; X, Y, Z = nf["grid_shape"]; vm = nf["voxel_mm"]
        out = math.sqrt(nf["a_indep"]) * torch.randn((n, self.V), generator=g, device=DEV)
        out += math.sqrt(nf["a_global"]) * torch.randn((n, 1), generator=g, device=DEV)
        for a, l in ((nf["a_short"], nf["l_short_mm"]), (nf["a_long"], nf["l_long_mm"])):
            ks, var = [], []
            for ax, (L, mm) in enumerate(zip((X, Y, Z), vm)):
                sig = l / math.sqrt(2) / mm; r = int(math.ceil(3 * sig)); xx = torch.arange(-r, r + 1, device=DEV).float()
                k = torch.exp(-xx ** 2 / (2 * sig ** 2)); ks.append(k)
                ones = torch.ones(1, 1, L, device=DEV); var.append(torch.nn.functional.conv1d(ones, (k ** 2)[None, None], padding=r)[0, 0])
            vv = var[0][self.ijk[:, 0]] * var[1][self.ijk[:, 1]] * var[2][self.ijk[:, 2]]
            for i in range(0, n, 16):
                c = min(16, n - i); w = torch.randn((c, 1, X, Y, Z), generator=g, device=DEV)
                w = torch.nn.functional.conv3d(w, ks[0][None, None, :, None, None], padding=(len(ks[0]) // 2, 0, 0))
                w = torch.nn.functional.conv3d(w, ks[1][None, None, None, :, None], padding=(0, len(ks[1]) // 2, 0))
                w = torch.nn.functional.conv3d(w, ks[2][None, None, None, None, :], padding=(0, 0, len(ks[2]) // 2))
                out[i:i + c] += math.sqrt(a) * w[:, 0, self.ijk[:, 0], self.ijk[:, 1], self.ijk[:, 2]] / vv.sqrt()
        return out
    def load(self, cond):
        t = time.time(); db = DB[cond]
        sig = torch.from_numpy(np.sqrt((10 ** (db / 10) - 1) * np.clip(self.var_n, 0, 1)).astype(np.float32)).to(DEV)
        for key, (s0, n) in self.blocks.items():
            x = torch.from_numpy(np.ascontiguousarray(self.mm[s0:s0 + n])).to(DEV)
            if db > 0:
                g = torch.Generator(device=DEV); g.manual_seed(seed_of(self.subj, cond, key))
                eps = self.corr_noise(n, g) if cond.endswith("c") else torch.randn(x.shape, generator=g, device=DEV)
                x = x.float() + sig * eps
                x = (x - x.mean(0)) / x.std(0).clamp_min(1e-6)
            self.R[s0:s0 + n] = x.half()
        sync(); return time.time() - t

def ranges_for(D, stories): return [D.blocks[s] for s in stories]

def chunks(ranges, ch=4096):
    for s0, n in ranges:
        for i in range(s0, s0 + n, ch): yield i, min(i + ch, s0 + n)

@torch.no_grad()
def pca_fit(D, ranges, q, niter=2, p=64, seed=0):
    R = D.R; V = D.V; n = sum(r[1] for r in ranges); k = min(q + p, n - 1)
    mu = torch.zeros(V, device=DEV)
    for a, b in chunks(ranges): mu += R[a:b].float().sum(0)
    mu /= n
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    Om = torch.randn(V, k, generator=g, device=DEV)
    def RX(M):   # (R - mu) @ M  -> n x k
        out = torch.empty(n, M.shape[1], device=DEV); c = 0; muM = mu @ M
        for a, b in chunks(ranges): out[c:c + b - a] = R[a:b].float() @ M - muM; c += b - a
        return out
    def RtY(Y):  # (R - mu)^T @ Y -> V x k
        out = torch.zeros(V, Y.shape[1], device=DEV); c = 0
        for a, b in chunks(ranges): out += R[a:b].float().T @ Y[c:c + b - a]; c += b - a
        return out - mu[:, None] * Y.sum(0)[None]
    Y = RX(Om); del Om
    for _ in range(niter):
        Qm, _ = torch.linalg.qr(Y); Z, _ = torch.linalg.qr(RtY(Qm)); Y = RX(Z)
    Qm, _ = torch.linalg.qr(Y)
    B = RtY(Qm).T                                    # k x V
    _, S, Vt = torch.linalg.svd(B, full_matrices=False)
    qq = min(q, k)
    return mu, Vt[:qq].contiguous(), S[:qq]

@torch.no_grad()
def project(D, mu, comps):
    F = torch.empty(D.N, comps.shape[0], device=DEV); C = comps.T.contiguous(); muC = mu @ C
    for a in range(0, D.N, 4096):
        b = min(a + 4096, D.N); F[a:b] = D.R[a:b].float() @ C - muC
    return F

def feats(F, s0, n, anchors):
    """anchors: LongTensor of anchor indices within block; returns (len, len(OFFS)*q)"""
    rows = anchors[:, None] + torch.tensor(OFFS, device=DEV)[None]
    valid = (rows >= 0) & (rows < n)
    X = F[(s0 + rows.clamp(0, n - 1)).reshape(-1)].reshape(len(anchors), len(OFFS), -1) * valid[..., None]
    return X.reshape(len(anchors), -1)

class Text:
    def __init__(self):
        t6 = np.load(f"{TXT}/train6.npz"); self.stories = list(t6["stories"])
        self.sid = {s: i for i, s in enumerate(self.stories)}
        self.tr_story = t6["story"]; self.tr_anchor = t6["anchor"]; self.tr_e = torch.tensor(t6["e"]).float().to(DEV)
        w6 = np.load(f"{TXT}/win6.npz")
        self.w6 = {k: w6[k] for k in ["story", "anchor", "nw", "foil_d", "foil_valid", "foil_oov", "true_oov"]}
        self.w6_e = torch.tensor(w6["e_st"]).float().to(DEV)
        self.w6_fe = torch.tensor(w6["foil_e"]).to(DEV)            # (N6, K6, 768) fp16
        self.names6 = list(w6["foil_names"])
        s20 = np.load(f"{TXT}/seg20.npz")
        self.s20 = {k: s20[k] for k in ["story", "anchor", "nw", "foil_d", "foil_valid", "foil_oov", "true_oov", "sub_mask"]}
        self.names20 = list(s20["foil_names"])
        self.sub = np.load(f"{TXT}/seg20_sub.npy", mmap_mode="r")    # (N20, 1+K20, 10, 768) fp16
        self.s20_true_sub = torch.tensor(np.ascontiguousarray(self.sub[:, 0])).to(DEV)   # (N20,10,768)
        self.s20_mask = torch.tensor(self.s20["sub_mask"]).to(DEV)
        self.ev6 = self.w6["nw"] >= 3; self.ev20 = self.s20["nw"] >= 3
    def train_sel(self, stories):
        ids = np.array([self.sid[s] for s in stories]); m = np.isin(self.tr_story, ids)
        return np.where(m)[0]

class Model(torch.nn.Module):
    def __init__(self, din):
        super().__init__(); self.lin = torch.nn.Linear(din, 768); torch.nn.init.normal_(self.lin.weight, std=0.1 / math.sqrt(din)); torch.nn.init.zeros_(self.lin.bias)
        self.ls = torch.nn.Parameter(torch.tensor(math.log(20.0)))
    def forward(self, x): return torch.nn.functional.normalize(self.lin(x), dim=-1)

def info_nce(z, e, st, an, scale):
    lg = scale * z @ e.T
    m = (st[:, None] == st[None]) & ((an[:, None] - an[None]).abs() <= 3)
    m.fill_diagonal_(False); lg = lg.masked_fill(m, -1e4)
    tgt = torch.arange(len(z), device=DEV)
    return 0.5 * (torch.nn.functional.cross_entropy(lg, tgt) + torch.nn.functional.cross_entropy(lg.T, tgt))

def build_xy(F, D, T, stories):
    Xs, Es, Ss, As = [], [], [], []
    for s in stories:
        s0, n = D.blocks[s]; sel = np.where(T.tr_story == T.sid[s])[0]
        an = torch.tensor(T.tr_anchor[sel], device=DEV)
        Xs.append(feats(F, s0, n, an)); Es.append(T.tr_e[sel]); Ss.append(torch.full((len(sel),), T.sid[s], device=DEV)); As.append(an)
    return torch.cat(Xs), torch.cat(Es), torch.cat(Ss), torch.cat(As)

def train(X, E, S, A, Xv, Ev, Sv, Av, lam, max_steps=4000, eval_every=None, patience=15, bs=1024, lr=3e-4, seed=0):
    """step-based (so small rungs get as many optimiser steps as large ones); early stopping on calibration loss"""
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    Xn = (X - mu) / sd; Xvn = (Xv - mu) / sd
    torch.manual_seed(seed); m = Model(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    best, best_state, best_step, bad, n_eval = 1e9, None, 0, 0, 0
    n = len(Xn); bs = min(bs, n); perm = torch.randperm(n, device=DEV); pi = 0
    if eval_every is None: eval_every = int(min(50, max(5, n // bs)))   # about once per epoch
    for step in range(1, max_steps + 1):
        if pi + bs > n: perm = torch.randperm(n, device=DEV); pi = 0
        b = perm[pi:pi + bs]; pi += bs
        m.train()
        loss = info_nce(m(Xn[b]), E[b], S[b], A[b], m.ls.exp().clamp(max=100)) + lam * (m.lin.weight ** 2).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % eval_every == 0:
            m.eval(); n_eval += 1
            with torch.no_grad(): vl = float(info_nce(m(Xvn), Ev, Sv, Av, m.ls.exp().clamp(max=100)))
            if vl < best - 1e-4: best, best_step, bad = vl, step, 0; best_state = {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if best_state is not None: m.load_state_dict(best_state)
    else: best = float("nan")
    return m, (mu, sd), best, best_step, step

class Scorer:
    """scores a fitted model on one block (story or test repeat)"""
    def __init__(self, D, T, F, m, norm):
        self.D, self.T, self.F, self.m, self.norm = D, T, F, m, norm
    @torch.no_grad()
    def z_block(self, key):
        s0, n = self.D.blocks[key]; an = torch.arange(0, n - 4, device=DEV)
        X = (feats(self.F, s0, n, an) - self.norm[0]) / self.norm[1]
        return self.m(X)                                            # (n-4, 768), row = anchor
    @torch.no_grad()
    def score6(self, key, story, shift_seed):
        T = self.T; sid = T.sid[story]; Z = self.z_block(key)
        sel = np.where((T.w6["story"] == sid) & T.ev6)[0]
        an = torch.tensor(T.w6["anchor"][sel], device=DEV)
        nA = Z.shape[0]; rng = np.random.default_rng(shift_seed); sh = int(rng.integers(15, max(16, nA - 15)))
        out = {}
        for tag, Zs in (("", Z[an]), ("_sh", Z[(an + sh) % nA])):
            st = (Zs * T.w6_e[sel]).sum(-1)
            sf = torch.einsum("nd,nkd->nk", Zs, T.w6_fe[sel].float())
            pool_m = torch.tensor(T.ev6 & (T.w6["story"] != sid), device=DEV)
            Pe = T.w6_e[pool_m]
            ps = Zs @ Pe.T
            rank = 1 + (ps > st[:, None]).sum(1)
            out["s_true" + tag] = st; out["s_foil" + tag] = sf; out["pool_scores" + tag] = ps; out["rank2" + tag] = rank
        out["sel"] = sel; out["npool"] = int(pool_m.sum())
        return out
    @torch.no_grad()
    def score20(self, key, story, shift_seed):
        T = self.T; sid = T.sid[story]; Z = self.z_block(key)
        sel = np.where((T.s20["story"] == sid) & T.ev20)[0]
        a = torch.tensor(T.s20["anchor"][sel], device=DEV)
        nA = Z.shape[0]; rng = np.random.default_rng(shift_seed + 1); sh = int(rng.integers(15, max(16, nA - 15)))
        sub = torch.tensor(np.ascontiguousarray(T.sub[sel])).to(DEV).float()    # (n,1+K,10,768)
        mask = T.s20_mask[sel].float()                                           # (n,10)
        pool_m = torch.tensor(T.ev20 & (T.s20["story"] != sid), device=DEV)
        Pe = T.s20_true_sub[pool_m].float(); Pm = T.s20_mask[pool_m].float()    # (P,10,768),(P,10)
        out = {}
        for tag, s in (("", 0), ("_sh", sh)):
            bidx = (a[:, None] + torch.arange(-9, 1, device=DEV)[None] + s) % nA    # (n,10)
            Zs = Z[bidx]                                                             # (n,10,768)
            sc = (torch.einsum("nbd,nkbd->nkb", Zs, sub) * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1)
            ps = torch.einsum("nbd,pbd->npb", Zs, Pe)
            ps = (ps * Pm[None]).sum(-1) / Pm.sum(-1).clamp_min(1)[None]
            st = sc[:, 0]; rank = 1 + (ps > st[:, None]).sum(1)
            out["s_true" + tag] = st; out["s_foil" + tag] = sc[:, 1:]; out["pool_scores" + tag] = ps; out["rank2" + tag] = rank
        out["sel"] = sel; out["npool"] = int(pool_m.sum())
        return out

def lse(x, dim=-1): return torch.logsumexp(x, dim)

def fit_T_theta(o, valid):
    """o: score dict on calibration; valid: (n,K) bool. returns T (bits temperature), theta (yes/no threshold)"""
    st, sf = o["s_true"], o["s_foil"]; vm = torch.tensor(valid, device=DEV)
    best = (-1e9, None)
    for T in np.logspace(-3, 0, 61):
        lg = torch.cat([st[:, None], sf.masked_fill(~vm, -1e4)], 1) / T
        ll = float((st / T - lse(lg)).mean())
        if ll > best[0]: best = (ll, T)
    Tb = best[1]
    pos = st.cpu().numpy(); neg = sf[vm].cpu().numpy()
    cand = np.quantile(np.concatenate([pos, neg]), np.linspace(0.001, 0.999, 999))
    ba = [(np.mean(pos > c) + np.mean(neg <= c)) / 2 for c in cand]
    th = float(cand[int(np.argmax(ba))])
    # cohort T-norm: per brain window, z-score against its scores for the other-story pool (candidate-independent)
    mu_c, sd_c = o["pool_scores"].mean(1, keepdim=True), o["pool_scores"].std(1, keepdim=True).clamp_min(1e-6)
    pos_t = ((st - mu_c[:, 0]) / sd_c[:, 0]).cpu().numpy(); neg_t = ((sf - mu_c) / sd_c)[vm].cpu().numpy()
    cand = np.quantile(np.concatenate([pos_t, neg_t]), np.linspace(0.001, 0.999, 999))
    ba_t = [(np.mean(pos_t > c) + np.mean(neg_t <= c)) / 2 for c in cand]
    th_t = float(cand[int(np.argmax(ba_t))])
    return float(Tb), th, float(np.max(ba)), th_t, float(np.max(ba_t))

def summarise(o, valid, dist, T, th):
    """per-window arrays for saving"""
    vm = torch.tensor(valid, device=DEV); r = {}
    for tag in ("", "_sh"):
        st, sf, ps = o["s_true" + tag], o["s_foil" + tag], o["pool_scores" + tag]
        lg1 = torch.cat([st[:, None], sf.masked_fill(~vm, -1e4)], 1) / T
        r["logp1" + tag] = (st / T - lse(lg1)).cpu().numpy() / math.log(2)
        lg2 = torch.cat([st[:, None], ps], 1) / T
        r["logp2" + tag] = (st / T - lse(lg2)).cpu().numpy() / math.log(2)
        r["s_true" + tag] = st.cpu().numpy().astype(np.float32); r["s_foil" + tag] = sf.cpu().numpy().astype(np.float16)
        r["rank2" + tag] = o["rank2" + tag].cpu().numpy().astype(np.int32)
        mu_c, sd_c = ps.mean(1), ps.std(1).clamp_min(1e-6)
        r["s_true_tn" + tag] = ((st - mu_c) / sd_c).cpu().numpy().astype(np.float32)
        r["s_foil_tn" + tag] = ((sf - mu_c[:, None]) / sd_c[:, None]).cpu().numpy().astype(np.float16)
    r["sel"] = o["sel"].astype(np.int32); r["npool"] = o["npool"]
    return r

def run_subject(subj, grid, out_dir, save_full_models=True):
    D = Data(subj); T = Text(); t_all = time.time()
    timings = {"subject": subj, "cells": {}}
    res = {}
    if os.path.exists(f"{out_dir}/{subj}_cells.json"): res = json.load(open(f"{out_dir}/{subj}_cells.json"))          # merge with earlier runs
    if os.path.exists(f"{out_dir}/{subj}_timings.json"): timings = json.load(open(f"{out_dir}/{subj}_timings.json")); timings.setdefault("cells", {})
    for cond in sorted(set(c for c, _ in grid), key=lambda c: DB[c]):
        tl = D.load(cond); timings["cells"][f"load_{cond}"] = tl
        for c2, k in grid:
            if c2 != cond: continue
            cell = f"{cond}_r{k}"; t_cell = time.time(); gpu = {"pca": 0.0, "train": 0.0, "eval": 0.0, "tune": 0.0}
            rung = [s for s in D.train_stories if D.sess[s] in SESS[:k]]
            hours = sum(D.blocks[s][1] for s in rung) * 2 / 3600
            Xv_key = "fromboyhoodtofatherhood"
            # ---------- tuning on the calibration story (full rung model)
            trials = []
            for q in QS:
                t = time.time(); mu, comps, _ = pca_fit(D, ranges_for(D, rung), q, seed=seed_of(subj, cond, k, "pca")); F = project(D, mu, comps); sync(); gpu["tune"] += time.time() - t
                X, E, S, A = build_xy(F, D, T, rung); Xv, Ev, Sv, Av = build_xy(F, D, T, [Xv_key])
                for lam in LAMS:
                    t = time.time(); m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, k, q, lam)); sync(); gpu["tune"] += time.time() - t
                    trials.append({"q": q, "lam": lam, "val_loss": vl, "best_step": bep, "steps": nep})
                    print(f"[{subj} {cell}] tune q={q} lam={lam:g} val={vl:.4f} step={bep}/{nep}", flush=True)
                del F, X
            bestt = min(trials, key=lambda x: x["val_loss"]); q, lam = bestt["q"], bestt["lam"]
            # ---------- models: full rung (outside stories + test + calib) and LOSO for inside stories
            models = [("FULL", None)] + [(s, s) for s in rung]
            cell_out = {"6": {}, "20": {}}; calib = {}
            for name, held in models:
                tr = [s for s in rung if s != held]
                t = time.time(); mu, comps, _ = pca_fit(D, ranges_for(D, tr), q, seed=seed_of(subj, cond, k, "pca", name)); F = project(D, mu, comps); sync(); gpu["pca"] += time.time() - t
                X, E, S, A = build_xy(F, D, T, tr); Xv, Ev, Sv, Av = build_xy(F, D, T, [Xv_key])
                t = time.time(); m, norm, vl, bep, nep = train(X, E, S, A, Xv, Ev, Sv, Av, lam, seed=seed_of(subj, cond, k, name)); sync(); gpu["train"] += time.time() - t
                del X
                t = time.time(); sc = Scorer(D, T, F, m, norm)
                # calibration: temperature + threshold for each readout
                o6c = sc.score6(Xv_key, Xv_key, seed_of(subj, cond, k, name, "c6")); o20c = sc.score20(Xv_key, Xv_key, seed_of(subj, cond, k, name, "c20"))
                T6, th6, ba6, tht6, bat6 = fit_T_theta(o6c, T.w6["foil_valid"][o6c["sel"]]); T20, th20, ba20, tht20, bat20 = fit_T_theta(o20c, T.s20["foil_valid"][o20c["sel"]])
                calib[name] = {"T6": T6, "th6": th6, "calib_ba6": ba6, "thtn6": tht6, "calib_ba_tn6": bat6, "T20": T20, "th20": th20, "calib_ba20": ba20,
                               "thtn20": tht20, "calib_ba_tn20": bat20, "val_loss": vl, "best_step": bep, "steps": nep}
                evals = [held] if held is not None else [s for s in D.train_stories if s not in rung] + [f"test_rep{r}" for r in range(10)]
                for ev in evals:
                    story = "wheretheressmoke" if ev.startswith("test_rep") else ev
                    o6 = sc.score6(ev, story, seed_of(subj, cond, k, ev, "6")); o20 = sc.score20(ev, story, seed_of(subj, cond, k, ev, "20"))
                    cell_out["6"][ev] = {**summarise(o6, T.w6["foil_valid"][o6["sel"]], None, T6, th6), "th": th6, "thtn": tht6, "T": T6, "model": name}
                    cell_out["20"][ev] = {**summarise(o20, T.s20["foil_valid"][o20["sel"]], None, T20, th20), "th": th20, "thtn": tht20, "T": T20, "model": name}
                sync(); gpu["eval"] += time.time() - t
                if save_full_models and name == "FULL" and cond == "native" and k == 15:
                    torch.save({"mu": mu.half().cpu(), "comps": comps.half().cpu(), "W": m.state_dict(), "norm": [x.cpu() for x in norm], "q": q, "lam": lam,
                                "T6": T6, "th6": th6, "T20": T20, "th20": th20, "thtn6": tht6, "thtn20": tht20}, f"{VP.WORK}/{subj}/full_native_model.pt")
                del F, sc, m
                print(f"[{subj} {cell}] model {name} val={vl:.3f} step={bep}/{nep} T6={T6:.3f} ba6c={ba6:.3f}/{bat6:.3f} ba20c={ba20:.3f}/{bat20:.3f} t={time.time()-t_cell:.0f}s", flush=True)
            n_params = len(OFFS) * q * 768 + 768 + 1
            res[cell] = {"cond": cond, "dB": -DB[cond], "rung_sessions": k, "hours": hours, "n_train_stories": len(rung), "q": q, "lam": lam,
                         "trials": trials, "calib": calib, "n_params_trainable": n_params, "n_params_pca": q * D.V,
                         "n_models": len(models), "gpu_s": gpu, "wall_s": time.time() - t_cell}
            np.savez_compressed(f"{out_dir}/{subj}_{cell}.npz", **flatten(cell_out))
            timings["cells"][cell] = {"gpu_s": gpu, "wall_s": time.time() - t_cell, "n_models": len(models), "n_trials": len(trials)}
            json.dump(res, open(f"{out_dir}/{subj}_cells.json", "w"), indent=1)
            json.dump(timings, open(f"{out_dir}/{subj}_timings.json", "w"), indent=1)
    timings["total_s"] = timings.get("total_s", 0) + time.time() - t_all
    json.dump(timings, open(f"{out_dir}/{subj}_timings.json", "w"), indent=1)

def flatten(cell_out):
    out = {}
    for rd, evs in cell_out.items():
        for ev, d in evs.items():
            for k, v in d.items():
                if k == "model": v = np.array(v)
                out[f"{rd}|{ev}|{k}"] = np.asarray(v)
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--cells", default="all")
    ap.add_argument("--out", default=f"{VP.WORK}/results"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    grid = GRID if a.cells == "all" else [g for g in GRID if f"{g[0]}_r{g[1]}" in a.cells.split(",")]
    run_subject(a.subject, grid, a.out)
