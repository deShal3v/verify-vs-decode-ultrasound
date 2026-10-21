"""Phase B: refit Tang 2023's encoding model (decoding/train_EM.py) and word-rate model (train_WR.py) for one cell
(subject x SNR condition x training rung) on the GPU.

Same algorithm as the repository, ported to torch:
  * stimulus features: Tang's own get_stim (GPT-1 'perceived', layer 9, 5-word context, Lanczos to TRs, z-scored
    with the rung's tr_stats, delays 1-4 TR); word rates: Tang's DataSequence.chunksums('lanczos', window=3);
  * bootstrap_ridge(use_corr=False): 50 bootstraps, chunks of 40 TRs, nchunks = ceil(T/5/40) held out, alphas
    logspace(1,3,10), per-voxel alpha = argmax of the bootstrap-mean R^2, voxel score = max over alphas of the
    bootstrap-mean R^2, top 10,000 voxels; final weights = ridge on all rows with each voxel's alpha;
  * noise model: for each training story, ridge on the other stories (per-voxel alphas, selected voxels), residual
    covariance on the held-out story, normalised by its mean diagonal, averaged over stories;
  * word-rate model (auditory ROI, the one run_decoder.py uses for perceived speech): bootstrap_ridge of the word
    rate on the ROI response delayed by -4..-1 TR; mean rate subtracted.
Ridge is solved through the Gram matrix, (X'X + a^2 I)^-1 X'Y, which equals the repository's SVD form
V diag(S/(S^2+a^2)) U'Y exactly (Tang's alphas are squared in both). Eigen-decompositions in float64, products in
float32. Chunk shuffles use a seeded random.Random (the repository's random.shuffle is unseeded).

Responses: the subject's ds003020 hf5 files (float32). Noise conditions add Phase A's noise exactly as verify.py
does (Data.corr_noise: fitted full-rank spatially correlated noise, same per-block seeds, per-voxel variance from
the test-story ncsnr), then re-z-score each block (training stories and the decoded test trial alike).

Outputs in --out: encoding_model.npz (compact: weights for the selected voxels only), word_rate_model_auditory.npz,
test_resp.npy (the decoded single trial = test repeat 0 with this cell's noise), pred_times.npz (word and TR times
predicted from test_resp, as run_decoder.py does), loso.npz (per held-out story residual covariances are NOT kept;
only the noise model), fit_info.json (timings, voxel/alpha summaries).
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, math, random, argparse, hashlib, itertools
import numpy as np, h5py, torch
REPO = VP.TANG_CODE; sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from GPT import GPT
from StimulusModel import LMFeatures
from utils_stim import get_stim, get_story_wordseqs, predict_word_rate, predict_word_times
from utils_ridge.DataSequence import DataSequence
from utils_ridge.util import make_delayed
from verify import Data, seed_of, DB as DB_A
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = False   # ridge in full fp32 (cudnn TF32 left as in Phase A for the noise generator)
SESS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
UTS = {"S1": "UTS01", "S2": "UTS02", "S3": "UTS03"}
TEST = "wheretheressmoke"
DB = dict(DB_A); DB.update({"m3c": 3.0, "m18c": 18.0, "m24c": 24.0})

def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)

class Noise:
    """Phase A noise generator (verify.Data.corr_noise / Data.load), without loading the response matrix to GPU"""
    def __init__(self, subj):
        d = f"{VP.WORK}/{subj}"; self.subj = subj
        D = Data.__new__(Data); D.subj = subj
        info = json.load(open(f"{d}/index.json")); D.V = info["V"]
        D.nf = json.load(open(f"{d}/noise_fit.json")); D.ijk = torch.tensor(np.load(f"{d}/vox_ijk.npy").astype(np.int64), device=DEV)
        self.D = D; self.var_n = np.load(f"{d}/var_noise.npy")
    def apply(self, x, cond, key):
        """x: (n, V) float32 numpy z-scored block -> noisy re-z-scored block (float32 cuda tensor), as Data.load"""
        db = DB[cond]
        if db <= 0: return torch.from_numpy(x).to(DEV)
        sig = torch.from_numpy(np.sqrt((10 ** (db / 10) - 1) * np.clip(self.var_n, 0, 1)).astype(np.float32)).to(DEV)
        g = torch.Generator(device=DEV); g.manual_seed(seed_of(self.subj, cond, key))
        xt = torch.from_numpy(x).to(DEV).float()
        eps = self.D.corr_noise(x.shape[0], g) if cond.endswith("c") else torch.randn(xt.shape, generator=g, device=DEV)
        xt = xt + sig * eps
        xt = (xt - xt.mean(0)) / xt.std(0).clamp_min(1e-6)
        del eps; return xt

def load_story(subj, story):
    with h5py.File(f"{VP.DS}/preprocessed_data/{UTS[subj]}/{story}.hf5", "r") as f:
        return np.nan_to_num(f["data"][:]).astype(np.float32)

def load_test_rep0(subj):
    with h5py.File(f"{VP.DS}/preprocessed_data/{UTS[subj]}/{TEST}.hf5", "r") as f:
        return np.nan_to_num(f["individual_repeats"][0]).astype(np.float32)

# ---------------------------------------------------------------- ridge (Gram form)
def eig_gram(G64):
    lam, V = torch.linalg.eigh(G64)
    return lam.clamp_min(0), V

def chunk_shuffle(nresp, chunklen, nchunks, rng):
    allinds = range(nresp)
    indchunks = list(zip(*[iter(allinds)] * chunklen))
    rng.shuffle(indchunks)
    held = list(itertools.chain(*indchunks[:nchunks]))
    return np.array(sorted(held))

def bootstrap_ridge_gpu(X, Y, alphas, nboots, chunklen, nchunks, seed, vchunk=8192):
    """Tang's bootstrap_ridge (use_corr=False) on GPU. X: (T,F) float32 cuda; Y: (T,V) float32 cuda.
    Multi-output (V>1): eigen-decomposition of the training Gram matrix (float64) per bootstrap.
    Single output (V==1, the word-rate model, F up to ~15k): Cholesky solves per alpha (float32 + 2 float64
    refinement steps). Returns per-column best alphas and the bootstrap-mean R^2 (A, V)."""
    T, F = X.shape; V = Y.shape[1]; A = len(alphas)
    a2 = torch.tensor(np.asarray(alphas, np.float64) ** 2, device=DEV)
    G = gram64(X)
    if V > 1: XtY_all = X.T @ Y
    else: XtY_all = sum(X[i:i + 4096].double().T @ Y[i:i + 4096].double() for i in range(0, T, 4096))
    rng = random.Random(seed)
    acc = torch.zeros(A, V, dtype=torch.float64, device=DEV)
    for bi in range(nboots):
        held = torch.from_numpy(chunk_shuffle(T, chunklen, nchunks, rng)).to(DEV)
        Xte = X[held]; Gtr = G - Xte.double().T @ Xte.double()
        if V > 1:
            lam, Vv = eig_gram(Gtr); Vf = Vv.float(); XteV = Xte @ Vf
            for c0 in range(0, V, vchunk):
                c1 = min(V, c0 + vchunk)
                Yte = Y[held, c0:c1]; B = Vf.T @ (XtY_all[:, c0:c1] - Xte.T @ Yte)
                var = Yte.var(0, unbiased=False)
                for ai in range(A):
                    Dg = (1.0 / (lam + a2[ai])).float()
                    pred = XteV @ (Dg[:, None] * B)
                    resvar = (Yte - pred).var(0, unbiased=False)
                    r = torch.nan_to_num((1 - resvar / var).clamp(0, 1), nan=0.0)
                    acc[ai, c0:c1] += r.double()
                del Yte, B
        else:
            Yte = Y[held].double(); b = XtY_all - Xte.double().T @ Yte; var = Yte.var(0, unbiased=False)
            for ai in range(A):
                w = chol_solve(Gtr, a2[ai], b)
                pred = Xte.double() @ w
                resvar = (Yte - pred).var(0, unbiased=False)
                acc[ai] += torch.nan_to_num((1 - resvar / var).clamp(0, 1), nan=0.0)
        if bi % 10 == 0: log(f"  boot {bi}/{nboots}")
    mean_sc = (acc / nboots).float()
    best = torch.argmax(mean_sc, 0)
    return np.asarray(alphas)[best.cpu().numpy()], mean_sc

def chol_solve(G64, a2, b64, refine=2):
    """solve (G + a2 I) w = b; float32 Cholesky + float64 iterative refinement"""
    M64 = G64 + a2 * torch.eye(G64.shape[0], dtype=torch.float64, device=G64.device)
    L = torch.linalg.cholesky(M64.float())
    w = torch.cholesky_solve(b64.float(), L).double()
    for _ in range(refine):
        r = b64 - M64 @ w
        w = w + torch.cholesky_solve(r.float(), L).double()
    return w

def gram64(X, ch=4096):
    """X'X in float64, accumulated over row chunks (bounded memory)"""
    return sum(X[i:i + ch].double().T @ X[i:i + ch].double() for i in range(0, X.shape[0], ch))

def ridge_weights_gpu(X, Y, valphas, G=None):
    """weights (F, V) for per-column alphas; X (T,F) cuda float32, Y (T,V) cuda float32"""
    if G is None: G = gram64(X)
    if Y.shape[1] == 1:
        b = sum(X[i:i + 4096].double().T @ Y[i:i + 4096].double() for i in range(0, X.shape[0], 4096))
        return chol_solve(G, float(np.asarray(valphas).ravel()[0]) ** 2, b).float()
    lam, Vv = eig_gram(G); Vf = Vv.float()
    B = Vf.T @ (X.T @ Y)
    a2 = torch.tensor(np.asarray(valphas, np.float64) ** 2, device=DEV)
    return Vf @ (B / (lam[:, None] + a2[None, :]).float())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--cond", default="native")
    ap.add_argument("--rung", type=int, default=15); ap.add_argument("--out", required=True)
    ap.add_argument("--nboots", type=int, default=config.NBOOTS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); T = {}
    s2s = json.load(open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")))
    stories = [st for s in SESS[:a.rung] for st in s2s[str(s)]]
    assert TEST not in stories
    log(f"{a.subject} {a.cond} r{a.rung}: {len(stories)} stories")
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: gpt_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    t = time.time()
    rstim, tr_stats, word_stats = get_stim(stories, features)
    stim_dict = {st: get_stim([st], features, tr_stats=tr_stats) for st in stories}
    T["stim_s"] = time.time() - t
    # ---- responses (+ noise)
    t = time.time(); noise = Noise(a.subject)
    test = noise.apply(load_test_rep0(a.subject), a.cond, "test_rep0").cpu().numpy()
    np.save(f"{a.out}/test_resp.npy", test)
    Tn = int(rstim.shape[0]); V = test.shape[1]
    Y = torch.empty((Tn, V), dtype=torch.float32, device=DEV); off = {}; r0 = 0
    for st in stories:
        y = noise.apply(load_story(a.subject, st), a.cond, st); n = y.shape[0]
        Y[r0:r0 + n] = y; off[st] = (r0, n); r0 += n; del y
    assert r0 == Tn, (r0, Tn)
    torch.cuda.empty_cache(); T["resp_s"] = time.time() - t
    log(f"stim {rstim.shape} resp ({Tn}, {V}) load {T['resp_s']:.0f}s")
    # ---- encoding model: bootstrap ridge over all voxels
    t = time.time()
    X = torch.from_numpy(rstim.astype(np.float32)).to(DEV)
    nchunks = int(np.ceil(Tn / 5 / config.CHUNKLEN))
    valphas, mean_sc = bootstrap_ridge_gpu(X, Y, config.ALPHAS, a.nboots, config.CHUNKLEN, nchunks, seed=seed_of(a.subject, a.cond, a.rung, "boot"))
    bscorrs = mean_sc.max(0).values.cpu().numpy(); del mean_sc
    vox = np.sort(np.argsort(bscorrs)[-config.VOXELS:])
    T["bootstrap_s"] = time.time() - t
    t = time.time()
    vox_t = torch.from_numpy(vox).to(DEV)
    Ysel = Y[:, vox_t].contiguous()
    W = ridge_weights_gpu(X, Ysel, valphas[vox])
    T["weights_s"] = time.time() - t
    rois = json.load(open(os.path.join(config.DATA_TRAIN_DIR, "ROIs", f"{a.subject}.json")))
    Yroi = {roi: Y[:, torch.from_numpy(np.array(rois[roi])).to(DEV)].cpu().numpy() for roi in ["auditory"]}
    del X, Y; torch.cuda.empty_cache()
    # ---- noise model (leave-one-story-out over the rung's stories)
    t = time.time()
    Xs = {st: torch.from_numpy(stim_dict[st].astype(np.float32)).to(DEV) for st in stories}
    Ys = {st: Ysel[off[st][0]:off[st][0] + off[st][1]] for st in stories}
    G_all = sum(Xs[st].double().T @ Xs[st].double() for st in stories)
    XtY_all = sum(Xs[st].T @ Ys[st] for st in stories)
    a2 = torch.tensor(np.asarray(valphas[vox], np.float64) ** 2, device=DEV)
    NM = torch.zeros(len(vox), len(vox), dtype=torch.float64, device=DEV)
    for i, h in enumerate(stories):
        if len(stories) == 1: break
        G = G_all - Xs[h].double().T @ Xs[h].double(); XtY = XtY_all - Xs[h].T @ Ys[h]
        lam, Vv = eig_gram(G); Vf = Vv.float()
        Wh = Vf @ ((Vf.T @ XtY) / (lam[:, None] + a2[None, :]).float())
        res = Ys[h] - Xs[h] @ Wh
        C = (res.T @ res).double()
        NM += C / torch.diagonal(C).mean() / len(stories)
    T["noise_model_s"] = time.time() - t
    NM = NM.cpu().numpy(); W = W.cpu().numpy(); del Xs, Ys, Ysel, G_all, XtY_all; torch.cuda.empty_cache()
    # ---- word-rate model (auditory ROI)
    t = time.time()
    wordseqs = get_story_wordseqs(stories)
    rates = {}
    for st in stories:
        ds = wordseqs[st]
        words = DataSequence(np.ones(len(ds.data_times)), ds.split_inds, ds.data_times, ds.tr_times)
        rates[st] = words.chunksums("lanczos", window=3)
    nz_rate = np.nan_to_num(np.concatenate([rates[st][5 + config.TRIM:-config.TRIM] for st in stories], axis=0)).reshape([-1, 1])
    mean_rate = float(np.mean(nz_rate)); rate = nz_rate - mean_rate
    wr = {}
    for roi in ["auditory"]:
        rv = np.array(rois[roi])
        delresp = make_delayed(Yroi[roi], config.RESP_DELAYS)
        Xw = torch.from_numpy(delresp.astype(np.float32)).to(DEV); Yw = torch.from_numpy(rate.astype(np.float32)).to(DEV)
        nch = int(np.ceil(Xw.shape[0] / 5 / config.CHUNKLEN))
        va, _ = bootstrap_ridge_gpu(Xw, Yw, config.ALPHAS, a.nboots, config.CHUNKLEN, nch, seed=seed_of(a.subject, a.cond, a.rung, "wr", roi))
        ww = ridge_weights_gpu(Xw, Yw, va).cpu().numpy().astype(np.float64)
        np.savez(f"{a.out}/word_rate_model_{roi}.npz", weights=ww, mean_rate=mean_rate, voxels=rv, alpha=va)
        wr[roi] = (ww, rv); del Xw
    T["word_rate_s"] = time.time() - t
    # ---- save encoding model (compact) + predicted test word times
    np.savez(f"{a.out}/encoding_model.npz", weights_sel=W.astype(np.float32), noise_model=NM,
             alphas_sel=valphas[vox], voxels=vox, stories=np.array(stories), tr_stats=np.array(tr_stats), word_stats=np.array(word_stats),
             bscorrs=bscorrs.astype(np.float32), alphas_all=valphas.astype(np.float32))
    word_rate = predict_word_rate(test, wr["auditory"][0], wr["auditory"][1], mean_rate)
    word_times, tr_times = predict_word_times(word_rate, test, starttime=-10)
    np.savez(f"{a.out}/pred_times.npz", word_times=word_times, tr_times=tr_times, word_rate=word_rate)
    info = {"subject": a.subject, "cond": a.cond, "dB": -DB[a.cond], "rung_sessions": a.rung, "n_stories": len(stories), "n_trs": int(Tn),
            "hours": Tn * 2 / 3600, "V": int(V), "nboots": a.nboots, "n_alpha_trials": len(config.ALPHAS),
            "params_encoding": int(W.shape[0] * W.shape[1]), "params_noise_model": int(len(vox) * (len(vox) + 1) // 2),
            "alpha_hist_sel": {str(x): int(np.sum(valphas[vox] == x)) for x in np.unique(valphas[vox])},
            "bscorr_sel_mean": float(bscorrs[vox].mean()), "bscorr_sel_min": float(bscorrs[vox].min()),
            "noise_model_diag_mean": float(np.diag(NM).mean()), "n_pred_words": int(len(word_times)),
            "wr_alpha": float(np.asarray(va).ravel()[0]),
            "timing_s": T, "total_s": time.time() - t0}
    json.dump(info, open(f"{a.out}/fit_info.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in info.items() if k != "alpha_hist_sel"}))

if __name__ == "__main__":
    main()
