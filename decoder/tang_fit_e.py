"""Phase E: tang_fit_us.py + --preset <npy of full-grid hf5 columns>: the model is fitted on exactly the preset voxels (intersected
with the condition's reachable set), all of them, no ranking by fit (DESIGN_E.md 1). Without --preset: identical to Phase C.
Original docstring: Phase C Tang encoding-model refit for one cell (DESIGN_C.md 5.1). Phase B tang_fit.py's algorithm unchanged
(get_stim, bootstrap ridge 50 x 10 alphas, chunk 40, top-N voxels, LOSO noise model), with these Phase C changes:
  * cell = <cond>_<rung>: rung r1/r4/r15 (sessions) or n1/n2/n3 (first stories); cond plain (Phase B noise, full grid)
    or ultrasound (noise_us.USNoise, voxel set restricted);
  * selected voxel count = min(10,000, n_set); saved 'voxels' are FULL-grid hf5 columns (so tang_decode / E-t-C index
    the full-grid response unchanged); test_resp.npy is full-grid with columns outside the set zeroed;
  * no word-rate model: pred_times.npz holds the FROZEN native 15.25 h predicted word times of the subject
    (--frozen = that cell's decoded.npz; Phase B nulls200 were generated from exactly these times).
Usage: python tang_fit_us.py --subject S1 --cell usL50_r15 --out $DATA_DIR/models/S1/usL50_r15 --frozen $WORK_DIR/S1/native_r15_decoded.npz"""
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tang_fit as TF
from tang_fit import config, GPT, LMFeatures, get_stim, predict_word_times, bootstrap_ridge_gpu, ridge_weights_gpu, eig_gram, load_story, load_test_rep0, log, DEV, TEST
from noise_us import USNoise, parse_cell, rung_stories, is_us, pseudo_db


def frozen_times(path, n_tr=291):
    """native 15.25 h predicted word times (decoded.npz['times']) + Tang's TR times for a 291-TR test response"""
    wt = np.load(path)["times"].astype(np.float64)
    _, tr_times = predict_word_times(np.zeros(n_tr, int), np.zeros((n_tr, 1)), starttime=-10)
    return wt, tr_times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--cell", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frozen", required=True); ap.add_argument("--nboots", type=int, default=config.NBOOTS)
    ap.add_argument("--preset", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    cond, rung = parse_cell(a.cell); t0 = time.time(); T = {}
    s2s = json.load(open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")))
    stories = rung_stories(rung, s2s); assert TEST not in stories
    log(f"{a.subject} {a.cell}: {len(stories)} stories")
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: gpt_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    t = time.time(); rstim, tr_stats, word_stats = get_stim(stories, features)
    stim_dict = {st: get_stim([st], features, tr_stats=tr_stats) for st in stories}; T["stim_s"] = time.time() - t
    # ---- responses + noise (training stories AND the decoded test trial; independent seeded draws per block)
    t = time.time(); noise = USNoise(a.subject)
    cols = noise.cols(cond)                                                    # None = full grid
    test_full = noise.apply(load_test_rep0(a.subject), cond, "test_rep0", restrict=False).cpu().numpy()
    np.save(f"{a.out}/test_resp.npy", test_full.astype(np.float32))
    Tn = int(rstim.shape[0]); Vset = test_full.shape[1] if cols is None else len(cols)
    Y = torch.empty((Tn, Vset), dtype=torch.float32, device=DEV); off = {}; r0 = 0
    for st in stories:
        y = noise.apply(load_story(a.subject, st), cond, st, restrict=True); n = y.shape[0]
        Y[r0:r0 + n] = y; off[st] = (r0, n); r0 += n; del y
    assert r0 == Tn, (r0, Tn); torch.cuda.empty_cache(); T["resp_s"] = time.time() - t
    if a.preset:                                                               # Phase E: the preset voxels, all of them
        PRE = np.load(a.preset).astype(np.int64); base = np.arange(Vset) if cols is None else cols
        pos = np.sort(np.where(np.isin(base, PRE))[0]); assert len(pos) > 0
        Y = Y[:, torch.from_numpy(pos).to(DEV)].contiguous(); cols = base[pos]; Vset = len(pos); torch.cuda.empty_cache()
        nvox = Vset
    else:
        nvox = min(config.VOXELS, Vset)
    log(f"stim {rstim.shape} resp ({Tn}, {Vset}) selecting {nvox}{' (preset, all)' if a.preset else ''}")
    # ---- encoding model (Phase B tang_fit.main, unchanged algorithm)
    t = time.time(); X = torch.from_numpy(rstim.astype(np.float32)).to(DEV)
    nchunks = int(np.ceil(Tn / 5 / config.CHUNKLEN))
    rk = int(rung[1:]) if rung[0] == "r" else rung                            # Phase B seed for session rungs (native_r15 = Phase B refit)
    valphas, mean_sc = bootstrap_ridge_gpu(X, Y, config.ALPHAS, a.nboots, config.CHUNKLEN, nchunks, seed=TF.seed_of(a.subject, cond, rk, "boot"))
    bscorrs = mean_sc.max(0).values.cpu().numpy(); del mean_sc
    vox = np.sort(np.argsort(bscorrs)[-nvox:]); T["bootstrap_s"] = time.time() - t
    t = time.time(); vox_t = torch.from_numpy(vox).to(DEV); Ysel = Y[:, vox_t].contiguous()
    W = ridge_weights_gpu(X, Ysel, valphas[vox]); T["weights_s"] = time.time() - t
    del X, Y; torch.cuda.empty_cache()
    t = time.time()
    Xs = {st: torch.from_numpy(stim_dict[st].astype(np.float32)).to(DEV) for st in stories}
    Ys = {st: Ysel[off[st][0]:off[st][0] + off[st][1]] for st in stories}
    G_all = sum(Xs[st].double().T @ Xs[st].double() for st in stories); XtY_all = sum(Xs[st].T @ Ys[st] for st in stories)
    a2 = torch.tensor(np.asarray(valphas[vox], np.float64) ** 2, device=DEV)
    NM = torch.zeros(len(vox), len(vox), dtype=torch.float64, device=DEV)
    for h in stories:
        if len(stories) == 1: break                                           # 1-story rung: Sigma = 0 (Phase B port behaviour)
        G = G_all - Xs[h].double().T @ Xs[h].double(); XtY = XtY_all - Xs[h].T @ Ys[h]
        lam, Vv = eig_gram(G); Vf = Vv.float(); Wh = Vf @ ((Vf.T @ XtY) / (lam[:, None] + a2[None, :]).float())
        res = Ys[h] - Xs[h] @ Wh; C = (res.T @ res).double(); NM += C / torch.diagonal(C).mean() / len(stories)
    T["noise_model_s"] = time.time() - t
    NM = NM.cpu().numpy(); W = W.cpu().numpy(); del Xs, Ys, Ysel, G_all, XtY_all; torch.cuda.empty_cache()
    vox_full = vox if cols is None else cols[vox]
    np.savez(f"{a.out}/encoding_model.npz", weights_sel=W.astype(np.float32), noise_model=NM, alphas_sel=valphas[vox], voxels=vox_full,
             stories=np.array(stories), tr_stats=np.array(tr_stats), word_stats=np.array(word_stats), bscorrs=bscorrs.astype(np.float32),
             alphas_all=valphas.astype(np.float32), set_cols=(np.arange(Vset) if cols is None else cols))
    wt, trt = frozen_times(a.frozen); np.savez(f"{a.out}/pred_times.npz", word_times=wt, tr_times=trt, frozen=True, source=a.frozen)
    info = {"subject": a.subject, "cell": a.cell, "cond": cond, "rung": rung, "dB": pseudo_db(cond), "ultrasound": is_us(cond),
            "n_stories": len(stories), "stories": stories, "n_trs": int(Tn), "hours": Tn * 2 / 3600, "V_set": int(Vset), "n_selected": int(len(vox)),
            "nboots": a.nboots, "n_alpha_trials": len(config.ALPHAS), "params_encoding": int(W.shape[0] * W.shape[1]),
            "params_noise_model": int(len(vox) * (len(vox) + 1) // 2), "alpha_hist_sel": {str(x): int(np.sum(valphas[vox] == x)) for x in np.unique(valphas[vox])},
            "bscorr_sel_mean": float(bscorrs[vox].mean()), "bscorr_sel_min": float(bscorrs[vox].min()), "noise_model_diag_mean": float(np.diag(NM).mean()),
            "noise_model_identity_1story": len(stories) == 1, "preset": a.preset, "n_preset_in_set": (int(Vset) if a.preset else None), "word_times": "frozen native_r15", "n_pred_words": int(len(wt)), "timing_s": T, "total_s": time.time() - t0}
    json.dump(info, open(f"{a.out}/fit_info.json", "w"), indent=1)
    log(json.dumps({k: v for k, v in info.items() if k not in ("alpha_hist_sel", "stories")}))


if __name__ == "__main__":
    main()
