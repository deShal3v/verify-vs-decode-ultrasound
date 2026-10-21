"""Phase B, encode-then-compare (E-t-C) verifier, step 2: score every Phase A window's candidates with one cell's
refitted Tang encoding model (same weights as the decoder = the 2x2 lower bound). DESIGN.md section 2.2.

score(c | window w) = -1/2 sum_{rows of w} (x_row(c) W - r_row)' P (x_row(c) W - r_row),  P = (Sigma(1-a) + aI)^-1, a = 2/3
x(c) = Tang's delayed, z-scored stimulus with the candidate's words at their times inside the window and 'blank'
(the rung's mean word embedding) at the true times of every other word of the story (no context).
Expanded exactly: score = <Ft_c, H'_w> - 1/2 Ft_c Q Ft_c + const_w, with Ft_c = F_c / sd (TR features of the
candidate, etc_feats.py), H'_w built from h_row = W P u_row (u = r - x0 W, x0 = all-blank stimulus) and the blank
term of the window, Q = the delay-expanded W P W'. Rows outside the story are dropped from the linear terms (the
quadratic term assumes all rows; only edge windows are affected).
Training-story windows: leave-one-story-out (W_-s by Gram-form ridge with the cell's per-voxel alphas on the
cell's voxels; Sigma_-s = (n Sigma - C_s/mean diag C_s)/(n-1)); other blocks (stories outside the rung, the
calibration story, the 10 test repeats): the full cell model. Noise: the cell's condition, Phase A seeds per block.
Output: Phase A verify.py format (results_etc/<subj>_<cell>.npz + <subj>_cells.json) so analyze.py metrics apply.

[etc_pool.py, Phase B deviation logged 2026-10-14 ~10:00Z] The cross-story pool and the T-norm cohort are TIMING-MATCHED:
the first E-t-C run (other-story windows at their own word times) showed the timing shortcut the review warned about
(20 s rank bits 9.9 vs the contrastive verifier's 5.2, while the timing-matched foils gave a LOWER yes/no accuracy). Here
each brain window w with n_w words at true times t_w is scored against 300 cross-story candidates of exactly n_w
consecutive words (random story != w's story, random start; drawn once per n with a fixed seed), placed at t_w; word
features within the candidate only (no context), as for the foils."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, math, gzip, argparse, numpy as np, torch, h5py
REPO = VP.TANG_CODE; sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim
from utils_ridge.interpdata import lanczosfun
from tang_fit import Noise, load_story, seed_of, UTS, DB, SESS
from prep_text import load_words
from verify import fit_T_theta, summarise, flatten
torch.backends.cuda.matmul.allow_tf32 = False
DEV = "cuda"; ALPHA = config.NM_ALPHA; TEST = "wheretheressmoke"; CALIB = "fromboyhoodtofatherhood"
SPAN = {"6": (-5, 3, -1, 4), "20": (-14, 3, -10, 4)}   # TR span lo,hi ; rows lo,hi (relative to anchor)

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

def lanczos_rows(times, trt):
    return np.stack([lanczosfun(0.5, x - times, 3) for x in trt]) if len(times) else np.zeros((len(trt), 0))

class Cell:
    def __init__(self, subj, cell_dir):
        em = np.load(f"{cell_dir}/encoding_model.npz")
        self.vox = em["voxels"]; self.W = torch.from_numpy(em["weights_sel"]).float().to(DEV)
        self.Sigma = torch.from_numpy(em["noise_model"]).double().to(DEV); self.alphas = em["alphas_sel"].astype(np.float64)
        self.mu = torch.from_numpy(em["tr_stats"][0]).float().to(DEV); self.sd = torch.from_numpy(em["tr_stats"][1]).float().to(DEV)
        self.blank = torch.from_numpy(em["word_stats"][0]).float().to(DEV); self.stories = list(em["stories"])

def precision(Sig):
    M = Sig * (1 - ALPHA) + ALPHA * torch.eye(Sig.shape[0], dtype=Sig.dtype, device=DEV)
    L = torch.linalg.cholesky(M.float()); return torch.cholesky_inverse(L)

def delay_expand(Z, rows, d=(1, 2, 3, 4)):
    """Z: (n_trs, 768) trimmed-TR features (index j); rows: LongTensor of row indices -> (len(rows), 3072);
    TR j<0 contributes zeros (Tang's per-story stimulus)"""
    out = []
    for dd in d:
        j = rows - dd; ok = (j >= 0) & (j < Z.shape[0])
        out.append(torch.where(ok[:, None], Z[j.clamp(0, Z.shape[0] - 1)], torch.zeros(1, Z.shape[1], device=DEV)))
    return torch.cat(out, 1)

class Scorer:
    def __init__(self, a):
        self.a = a; self.subj = a.subject
        self.stories_all, self.Wd = load_words()
        self.sid = {s: i for i, s in enumerate(self.stories_all)}
        t6 = np.load(f"{VP.WORK}/text/win6.npz"); t20 = np.load(f"{VP.WORK}/text/seg20.npz")
        self.tx = {"6": {k: t6[k] for k in ["story", "anchor", "nw", "foil_valid"]}, "20": {k: t20[k] for k in ["story", "anchor", "nw", "foil_valid"]}}
        self.F = {k: np.load(f"{VP.WORK}/etc/F{k}.npy", mmap_mode="r") for k in ("6", "20")}
        self.S = {k: torch.from_numpy(np.load(f"{VP.WORK}/etc/S{k}.npy")).float().to(DEV) for k in ("6", "20")}
        self.ev = {k: self.tx[k]["nw"] >= 3 for k in ("6", "20")}
        self.build_pools()
        self.noise = Noise(self.subj); self.kappa = 1.0

    def build_pools(self, K=300, K_draw=330):
        """per kind and word count n: K_draw candidate texts of n consecutive words from random stories (fixed seed)"""
        cache = f"{VP.WORK}/etc/pools_v1.pt"
        if os.path.exists(cache):
            d = torch.load(cache); self.pool = d["pool"]; self.K = K; return
        from etc_feats import make_stim_batch
        vocab = json.load(open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")))
        gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=vocab, device="cuda")
        rng = np.random.default_rng(20261014); st_names = self.stories_all; nwords = np.array([len(self.Wd[s][0]) for s in st_names])
        self.pool = {}; self.K = K
        for k in ("6", "20"):
            ev = self.tx[k]["nw"] >= 3; ns = np.unique(self.tx[k]["nw"][ev]); self.pool[k] = {}
            cands, meta = [], []
            for n in ns:
                ok = np.where(nwords > n)[0]; src = rng.choice(ok, K_draw); j0 = [int(rng.integers(0, nwords[i] - n)) for i in src]
                for i, j in zip(src, j0): cands.append(self.Wd[st_names[i]][0][j:j + n]); meta.append((int(n), int(i)))
            feats = make_stim_batch(gpt, cands)
            for n in ns:
                idx = [q for q, (nn, _) in enumerate(meta) if nn == n]
                self.pool[k][int(n)] = (torch.from_numpy(np.stack([feats[q] for q in idx]).astype(np.float16)), np.array([meta[q][1] for q in idx]))
        del gpt; torch.cuda.empty_cache()
        torch.save({"pool": self.pool, "K_draw": K_draw, "seed": 20261014}, cache + ".tmp"); os.replace(cache + ".tmp", cache)

    def pool_F(self, story, k, sel):
        """timing-matched pool features for the brain windows sel of this story: (m, K, L, 768) fp16 on GPU, by word count"""
        lo, hi, _, _ = SPAN[k]; L = hi - lo + 1; sid = self.sid[story]; times_all = np.asarray(self.Wd[story][1]); W_ = 6.0 if k == "6" else 20.0
        out = torch.empty((len(sel), self.K, L, 768), dtype=torch.float16, device=DEV)
        for r, wi in enumerate(sel):
            a = int(self.tx[k]["anchor"][wi]); T = 2 * a + 12.0; tt = times_all[(times_all > T - W_) & (times_all <= T)]
            E, src = self.pool[k][len(tt)]; keep = np.where(src != sid)[0][:self.K]
            Lm = torch.from_numpy(lanczos_rows(tt, 2.0 * (a + np.arange(lo, hi + 1)) + 11.0)).float().to(DEV)      # (L, n)
            out[r] = torch.einsum("ln,cnd->cld", Lm, E[keep].to(DEV).float()).half()
        return out

    def S_of(self, story, anchors, k):
        """sum_i lanczos(2j+11 - t_i) over the words of the window ending at 2a+12 (true times), on the span of anchor a"""
        lo, hi, _, _ = SPAN[k]; times = np.asarray(self.Wd[story][1]); W_ = 6.0 if k == "6" else 20.0
        out = np.zeros((len(anchors), hi - lo + 1), np.float32)
        for r, a in enumerate(anchors.tolist()):
            T = 2 * a + 12.0; tt = times[(times > T - W_) & (times <= T)]
            if len(tt): out[r] = lanczos_rows(tt, 2.0 * (a + np.arange(lo, hi + 1)) + 11.0).sum(1)
        return torch.from_numpy(out).to(DEV)

    def story_base(self, st, cell, n_rows):
        """all-blank z-scored TR features for story st's trimmed TRs 0..n_rows-1 (n_rows, 768)"""
        words, times, _ = self.Wd[st]
        trt = 2.0 * np.arange(n_rows) + 11.0
        Ssum = torch.from_numpy(lanczos_rows(np.asarray(times), trt).sum(1)).float().to(DEV)
        return (Ssum[:, None] * cell.blank[None] - cell.mu[None]) / cell.sd[None]

    def block_resp(self, key, cond):
        if key.startswith("test_rep"):
            with h5py.File(f"{VP.DS}/preprocessed_data/{UTS[self.subj]}/{TEST}.hf5", "r") as f:
                x = np.nan_to_num(f["individual_repeats"][int(key[8:])]).astype(np.float32)
        else: x = load_story(self.subj, key)
        return self.noise.apply(x, cond, key)

    def Q_apply(self, G, Ft, k):
        """quadratic form 1/2 sum_rows dx G dx for candidates Ft (m, L, 768), full rows"""
        lo, hi, r0, r1 = SPAN[k]; L = hi - lo + 1
        rows = torch.arange(r0, r1 + 1, device=DEV) - lo          # row index in span coordinates
        q = torch.zeros(Ft.shape[0], device=DEV)
        for c0 in range(0, Ft.shape[0], 2048):
            X = Ft[c0:c0 + 2048].float()
            dx = torch.cat([X[:, rows - d] for d in (1, 2, 3, 4)], 2)   # (m, nrows, 3072)
            q[c0:c0 + 2048] = 0.5 * ((dx @ G) * dx).sum((1, 2))
        return q

    def score_block(self, key, story, cell, W, P, cond, shift_seed):
        """returns {'6': dict, '20': dict} in verify.Scorer.score6/score20 format"""
        R = self.block_resp(key, cond)[:, torch.from_numpy(cell.vox).to(DEV)]          # (n, 10k)
        n = R.shape[0]
        Z0 = self.story_base(story, cell, n)
        rows_all = torch.arange(n, device=DEV)
        X0 = delay_expand(Z0, rows_all)                                                # (n, 3072)
        U = R - X0 @ W
        WP = W @ P                                                                     # (3072, 10k)
        h = U @ WP.T                                                                   # (n, 3072)
        G = WP @ W.T                                                                   # (3072, 3072)
        out = {}
        sid = self.sid[story]
        for k in ("6", "20"):
            lo, hi, r0, r1 = SPAN[k]; Lsp = hi - lo + 1
            tx = self.tx[k]; sel = np.where((tx["story"] == sid) & self.ev[k])[0]
            anchors = torch.from_numpy(tx["anchor"][sel]).long().to(DEV)
            nA = n - 4; rng = np.random.default_rng(shift_seed + (0 if k == "6" else 1)); sh = int(rng.integers(15, max(16, nA - 15)))
            Fw = torch.from_numpy(np.ascontiguousarray(self.F[k][sel])).to(DEV)          # (m, K1, L, 768) f16
            sdv = cell.sd
            def Hprime(anc):
                Sblank = self.S_of(story, anc, k)
                # linear vector per brain window in span coords: sum over rows/delays of h_row blocks, + blank cross term
                rr = anc[:, None] + torch.arange(r0, r1 + 1, device=DEV)[None]          # (m, nrows)
                valid = (rr >= 0) & (rr < n)
                hr = h[rr.clamp(0, n - 1)] * valid[..., None]                            # (m, nrows, 3072)
                # blank term: B (m, L, 768) = Sblank * blank / sd ; contributes -(dx_B) G to the linear vector (from expanding the square)
                B = Sblank[:, :, None] * (cell.blank / sdv)[None, None]
                rows = torch.arange(r0, r1 + 1, device=DEV) - lo
                dB = torch.cat([B[:, rows - d] for d in (1, 2, 3, 4)], 2)               # (m, nrows, 3072)
                v = (hr + dB @ G) * valid[..., None]                                     # (m, nrows, 3072): coefficient on dx(c)
                Hs = torch.zeros(anc.shape[0], Lsp, 768, device=DEV)
                for di, d in enumerate((1, 2, 3, 4)):
                    Hs.index_add_(1, rows - d, v[:, :, di * 768:(di + 1) * 768])
                absj = anc[:, None] + lo + torch.arange(Lsp, device=DEV)[None]          # Tang's per-story stimulus is zero before trimmed TR 0
                return Hs * (absj >= 0)[..., None]
            # candidate terms: Ft = F / sd ; score = <Ft, H'> - 1/2 Ft Q Ft  (blank cross term inside H')
            Ft = Fw.float() / sdv[None, None, None]
            m, K1 = Ft.shape[0], Ft.shape[1]
            q_c = self.Q_apply(G, Ft.reshape(m * K1, Lsp, 768), k).reshape(m, K1)
            Pw = self.pool_F(story, k, sel)                                                # (m, K, L, 768) timing-matched pool
            Pt = Pw.float() / sdv[None, None, None]
            q_p = self.Q_apply(G, Pt.reshape(m * self.K, Lsp, 768), k).reshape(m, self.K)
            res = {}
            for tag, anc_b in (("", anchors), ("_sh", (anchors + sh) % nA)):
                Hp = Hprime(anc_b)                                                       # brain window (shifted for _sh)
                lin_c = torch.einsum("mkld,mld->mk", Ft, Hp)
                sc = (lin_c - q_c) / self.kappa
                lin_p = torch.einsum("mkld,mld->mk", Pt, Hp)                              # (m, K)
                ps = (lin_p - q_p) / self.kappa
                st_ = sc[:, 0]; res["s_true" + tag] = st_; res["s_foil" + tag] = sc[:, 1:]; res["pool_scores" + tag] = ps
                res["rank2" + tag] = 1 + (ps > st_[:, None]).sum(1)
            res["sel"] = sel; res["npool"] = int(self.K)
            out[k] = res
            del Fw, Ft, Pw, Pt
        return out

def fold_models(cell, stim, resp_sel, stories):
    """yield (story, W_-s, P_-s) for LOSO over the cell's rung stories (Tang noise-model folds)"""
    a2 = torch.tensor(cell.alphas ** 2, device=DEV)
    G_all = sum(stim[s].double().T @ stim[s].double() for s in stories)
    XtY_all = sum(stim[s].T @ resp_sel[s] for s in stories)
    n = len(stories); NMs = cell.Sigma
    for h_ in stories:
        G = G_all - stim[h_].double().T @ stim[h_].double(); XtY = XtY_all - stim[h_].T @ resp_sel[h_]
        lam, Vv = torch.linalg.eigh(G); lam = lam.clamp_min(0); Vf = Vv.float()
        Wh = Vf @ ((Vf.T @ XtY) / (lam[:, None] + a2[None, :]).float())
        res = resp_sel[h_] - stim[h_] @ Wh; C = (res.T @ res).double(); C = C / torch.diagonal(C).mean()
        Sig = (n * NMs - C) / (n - 1) if n > 1 else NMs
        yield h_, Wh, precision(Sig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default=f"{VP.WORK}/results_etc2"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); cond, rk = a.cell.rsplit("_r", 1); rung = int(rk)
    cell_dir = f"{VP.MODELS}/{a.subject}/{a.cell}"; cell = Cell(a.subject, cell_dir); S = Scorer(a)
    s2s = json.load(open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")))
    all_train = [st for s in SESS for st in s2s[str(s)]]
    rung_st = cell.stories; assert rung_st == [st for s in SESS[:rung] for st in s2s[str(s)]]
    # stimulus per story (Tang get_stim with the cell's tr_stats) and responses of the rung stories (selected voxels)
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=vocab, device="cuda")
    feats = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    tr_stats = (cell.mu.double().cpu().numpy(), cell.sd.double().cpu().numpy())
    stim = {st: torch.from_numpy(get_stim([st], feats, tr_stats=tr_stats).astype(np.float32)).to(DEV) for st in rung_st}
    vox_t = torch.from_numpy(cell.vox).to(DEV)
    resp_sel = {st: S.block_resp(st, cond)[:, vox_t].contiguous() for st in rung_st}
    log(f"loaded {len(rung_st)} rung stories t={time.time()-t0:.0f}s")
    cell_out = {"6": {}, "20": {}}; calib = {}
    Pfull = precision(cell.Sigma)
    def run_block(ev, story, W, P, name, Tcal):
        o = S.score_block(ev, story, cell, W, P, cond, seed_of(a.subject, cond, rung, ev, "etc"))
        for k in ("6", "20"):
            T_, th, tht = Tcal[k]
            cell_out[k][ev] = {**summarise(o[k], S.tx[k]["foil_valid"][o[k]["sel"]], None, T_, th), "th": th, "thtn": tht, "T": T_, "model": name}
    # calibration (full model): temperature + thresholds, as Phase A
    oc = S.score_block(CALIB, CALIB, cell, cell.W, Pfull, cond, seed_of(a.subject, cond, rung, "calib", "etc"))
    # score scale: log-likelihood units -> divide by kappa so the calibration pool-score sd is ~0.05 (the cosine scale of
    # Phase A's temperature grid); ranks, 2AFC, T-norm and yes/no are invariant to this constant
    S.kappa = float(oc["6"]["pool_scores"].std(1).median()) / 0.05
    oc = S.score_block(CALIB, CALIB, cell, cell.W, Pfull, cond, seed_of(a.subject, cond, rung, "calib", "etc"))
    Tcal = {}
    for k in ("6", "20"):
        T_, th, ba, tht, bat = fit_T_theta(oc[k], S.tx[k]["foil_valid"][oc[k]["sel"]])
        Tcal[k] = (T_, th, tht); calib[k] = {"T": T_, "th": th, "calib_ba": ba, "thtn": tht, "calib_ba_tn": bat}
    log(f"calib {json.dumps(calib)}")
    for ev in [s for s in all_train if s not in rung_st] + [f"test_rep{r}" for r in range(10)]:
        run_block(ev, TEST if ev.startswith("test_rep") else ev, cell.W, Pfull, "FULL", Tcal)
    log(f"full-model blocks done t={time.time()-t0:.0f}s")
    for i, (h_, Wh, Ph) in enumerate(fold_models(cell, stim, resp_sel, rung_st)):
        run_block(h_, h_, Wh, Ph, h_, Tcal)
        if i % 10 == 0: log(f"fold {i}/{len(rung_st)} t={time.time()-t0:.0f}s")
    np.savez_compressed(f"{a.out}/{a.subject}_{a.cell}.npz", **flatten(cell_out))
    cj = f"{a.out}/{a.subject}_cells.json"; res = json.load(open(cj)) if os.path.exists(cj) else {}
    hours = sum(stim[s].shape[0] for s in rung_st) * 2 / 3600
    res[a.cell] = {"cond": cond, "dB": -DB[cond], "rung_sessions": rung, "hours": hours, "n_train_stories": len(rung_st), "q": None, "lam": None,
                   "trials": [{"alpha_grid": 10, "nboots": 50}], "calib": calib, "n_params_trainable": int(cell.W.numel()), "n_params_pca": 0,
                   "n_models": 1 + len(rung_st), "gpu_s": {"etc_total": time.time() - t0}, "wall_s": time.time() - t0, "readout": "encode-then-compare (timing-matched pool and cohort)", "kappa": S.kappa, "pool_K": S.K}
    json.dump(res, open(cj, "w"), indent=1)
    log(f"done {a.cell} t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
