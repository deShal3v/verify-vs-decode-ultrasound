"""Phase C encode-then-compare verifier (DESIGN_C.md 5.3): Phase B etc_pool.py (timing-matched cross-story pool and
T-norm cohort, LOSO folds with Tang's noise-model folds, calibration as Phase A) with these changes only:
  * pool K = 600 (was 300): K_draw = 700 per word count, seed 20261015, cache $WORK_DIR/etc/pools_v2_K600.pt; the
    first 600 candidates not from the window's own story are kept (asserted); rank-bits cap log2(601) = 9.23;
  * the pool is scored in window chunks (Phase B's -12 dB cells ran out of GPU memory);
  * cells <cond>_<rung> with ultrasound conditions (noise_us.USNoise, full grid with columns outside the set zeroed;
    the cell's encoding model indexes only set columns) and story rungs n1/n2/n3; a 1-story rung has no LOSO fold
    (Tang's noise covariance is 0 there, i.e. precision I/alpha);
  * one json per cell (parallel jobs never share a file).
Usage: python etc_pool600.py --subject S1 --cell usL50_r15"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import etc_pool as EP
from etc_pool import (config, GPT, LMFeatures, get_stim, lanczos_rows, Cell, precision, delay_expand, fold_models, fit_T_theta, summarise, flatten,
                      seed_of, SPAN, DEV, TEST, CALIB, log, load_story, UTS)
from noise_us import USNoise, parse_cell, rung_stories, is_us, pseudo_db
K_POOL, K_DRAW, POOL_SEED = 600, 700, 20261015
WCH = {"6": 32, "20": 16}                  # windows per pool chunk


class Scorer600(EP.Scorer):
    def __init__(self, a, cond):
        self.cond = cond; super().__init__(a); self.noise = USNoise(self.subj)

    def build_pools(self, K=K_POOL, K_draw=K_DRAW):
        cache = f"{VP.WORK}/etc/pools_v2_K600.pt"; self.K = K
        if os.path.exists(cache):
            d = torch.load(cache); assert d["K_draw"] == K_draw and d["seed"] == POOL_SEED; self.pool = d["pool"]; return
        from etc_feats import make_stim_batch
        vocab = json.load(open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")))
        gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=vocab, device="cuda")
        rng = np.random.default_rng(POOL_SEED); st_names = self.stories_all; nwords = np.array([len(self.Wd[s][0]) for s in st_names])
        self.pool = {}
        for k in ("6", "20"):
            ev = self.tx[k]["nw"] >= 3; ns = np.unique(self.tx[k]["nw"][ev]); self.pool[k] = {}
            cands, meta = [], []
            for n in ns:
                ok = np.where(nwords > n)[0]; src = rng.choice(ok, K_draw); j0 = [int(rng.integers(0, nwords[i] - n)) for i in src]
                for i, j in zip(src, j0): cands.append(self.Wd[st_names[i]][0][j:j + n]); meta.append((int(n), int(i)))
            # features per word count (2026-10-16 memory fix, same draws: make_stim_batch's (rows, ctx+1, 768) float32 buffer over ALL
            # candidates is ~50 GB at K_draw = 700 and OOM-killed / livelocked the 64 GB verifier nodes); per-list results are
            # independent of the other lists (fixed-length rows, no padding), so the pool is the same up to batching numerics
            for n in ns:
                idx = [q for q, (nn, _) in enumerate(meta) if nn == n]
                f = make_stim_batch(gpt, [cands[q] for q in idx])
                self.pool[k][int(n)] = (torch.from_numpy(np.stack(f).astype(np.float16)), np.array([meta[q][1] for q in idx])); del f
        del gpt; torch.cuda.empty_cache()
        torch.save({"pool": self.pool, "K_draw": K_draw, "seed": POOL_SEED, "K": K}, cache + ".tmp"); os.replace(cache + ".tmp", cache)

    def pool_F(self, story, k, sel):
        lo, hi, _, _ = SPAN[k]; L = hi - lo + 1; sid = self.sid[story]; times_all = np.asarray(self.Wd[story][1]); W_ = 6.0 if k == "6" else 20.0
        out = torch.empty((len(sel), self.K, L, 768), dtype=torch.float16, device=DEV)
        for r, wi in enumerate(sel):
            a = int(self.tx[k]["anchor"][wi]); T = 2 * a + 12.0; tt = times_all[(times_all > T - W_) & (times_all <= T)]
            E, src = self.pool[k][len(tt)]; keep = np.where(src != sid)[0][:self.K]; assert len(keep) == self.K, (story, k, len(tt), len(keep))
            Lm = torch.from_numpy(lanczos_rows(tt, 2.0 * (a + np.arange(lo, hi + 1)) + 11.0)).float().to(DEV)
            out[r] = torch.einsum("ln,cnd->cld", Lm, E[keep].to(DEV).float()).half()
        return out

    def block_resp(self, key, cond):
        import h5py
        if key.startswith("test_rep"):
            with h5py.File(f"{VP.DS}/preprocessed_data/{UTS[self.subj]}/{TEST}.hf5", "r") as f:
                x = np.nan_to_num(f["individual_repeats"][int(key[8:])]).astype(np.float32)
        else: x = load_story(self.subj, key)
        return self.noise.apply(x, cond, key, restrict=False)

    def score_block(self, key, story, cell, W, P, cond, shift_seed):
        """etc_pool.Scorer.score_block with the pool scored in window chunks (identical arithmetic)"""
        R = self.block_resp(key, cond)[:, torch.from_numpy(cell.vox).to(DEV)]
        n = R.shape[0]
        Z0 = self.story_base(story, cell, n)
        X0 = delay_expand(Z0, torch.arange(n, device=DEV))
        U = R - X0 @ W; WP = W @ P; h = U @ WP.T; G = WP @ W.T
        out = {}; sid = self.sid[story]
        for k in ("6", "20"):
            lo, hi, r0, r1 = SPAN[k]; Lsp = hi - lo + 1
            tx = self.tx[k]; sel = np.where((tx["story"] == sid) & self.ev[k])[0]
            anchors = torch.from_numpy(tx["anchor"][sel]).long().to(DEV)
            nA = n - 4; rng = np.random.default_rng(shift_seed + (0 if k == "6" else 1)); sh = int(rng.integers(15, max(16, nA - 15)))
            Fw = torch.from_numpy(np.ascontiguousarray(self.F[k][sel])).to(DEV); sdv = cell.sd
            def Hprime(anc):
                Sblank = self.S_of(story, anc, k)
                rr = anc[:, None] + torch.arange(r0, r1 + 1, device=DEV)[None]; valid = (rr >= 0) & (rr < n)
                hr = h[rr.clamp(0, n - 1)] * valid[..., None]
                B = Sblank[:, :, None] * (cell.blank / sdv)[None, None]
                rows = torch.arange(r0, r1 + 1, device=DEV) - lo
                dB = torch.cat([B[:, rows - d] for d in (1, 2, 3, 4)], 2)
                v = (hr + dB @ G) * valid[..., None]
                Hs = torch.zeros(anc.shape[0], Lsp, 768, device=DEV)
                for di, d in enumerate((1, 2, 3, 4)): Hs.index_add_(1, rows - d, v[:, :, di * 768:(di + 1) * 768])
                absj = anc[:, None] + lo + torch.arange(Lsp, device=DEV)[None]
                return Hs * (absj >= 0)[..., None]
            Ft = Fw.float() / sdv[None, None, None]; m, K1 = Ft.shape[0], Ft.shape[1]
            q_c = self.Q_apply(G, Ft.reshape(m * K1, Lsp, 768), k).reshape(m, K1)
            Hps = {tag: Hprime(anc_b) for tag, anc_b in (("", anchors), ("_sh", (anchors + sh) % nA))}
            ps = {tag: torch.empty((m, self.K), device=DEV) for tag in Hps}
            for c0 in range(0, m, WCH[k]):                                         # pool in window chunks
                c1 = min(m, c0 + WCH[k])
                Pt = self.pool_F(story, k, sel[c0:c1]).float() / sdv[None, None, None]
                q_p = self.Q_apply(G, Pt.reshape((c1 - c0) * self.K, Lsp, 768), k).reshape(c1 - c0, self.K)
                for tag, Hp in Hps.items():
                    ps[tag][c0:c1] = (torch.einsum("mkld,mld->mk", Pt, Hp[c0:c1]) - q_p) / self.kappa
                del Pt, q_p
            res = {}
            for tag, Hp in Hps.items():
                sc = (torch.einsum("mkld,mld->mk", Ft, Hp) - q_c) / self.kappa
                st_ = sc[:, 0]; res["s_true" + tag] = st_; res["s_foil" + tag] = sc[:, 1:]; res["pool_scores" + tag] = ps[tag]
                res["rank2" + tag] = 1 + (ps[tag] > st_[:, None]).sum(1)
            res["sel"] = sel; res["npool"] = int(self.K); out[k] = res
            del Fw, Ft
        return out


def build_pool_only():
    """build the subject-independent K=600 pool cache (needs only the text side, not the candidate features)"""
    from prep_text import load_words
    S = Scorer600.__new__(Scorer600); S.stories_all, S.Wd = load_words()
    t6 = np.load(f"{VP.WORK}/text/win6.npz"); t20 = np.load(f"{VP.WORK}/text/seg20.npz")
    S.tx = {"6": {k: t6[k] for k in ["story", "anchor", "nw", "foil_valid"]}, "20": {k: t20[k] for k in ["story", "anchor", "nw", "foil_valid"]}}
    S.build_pools(); n = {k: {int(nn): int(v[0].shape[0]) for nn, v in S.pool[k].items()} for k in S.pool}
    log(f"pool cache built: K_draw per word count min {min(min(v.values()) for v in n.values())}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--subject", required=True); ap.add_argument("--cell", required=True)
    ap.add_argument("--out", default=f"{VP.WORK}/results_etc600"); ap.add_argument("--build_pool_only", action="store_true"); a = ap.parse_args()
    if a.build_pool_only: build_pool_only(); return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time(); cond, rung = parse_cell(a.cell)
    cell_dir = f"{VP.MODELS}/{a.subject}/{a.cell}"; cell = Cell(a.subject, cell_dir); S = Scorer600(a, cond)
    s2s = json.load(open(os.path.join(config.DATA_TRAIN_DIR, "sess_to_story.json")))
    all_train = [st for s in EP.SESS for st in s2s[str(s)]]
    rung_st = cell.stories; assert rung_st == rung_stories(rung, s2s), (rung_st, rung)
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=vocab, device="cuda")
    feats = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    tr_stats = (cell.mu.double().cpu().numpy(), cell.sd.double().cpu().numpy())
    stim = {st: torch.from_numpy(get_stim([st], feats, tr_stats=tr_stats).astype(np.float32)).to(DEV) for st in rung_st}
    vox_t = torch.from_numpy(cell.vox).to(DEV)
    resp_sel = {st: S.block_resp(st, cond)[:, vox_t].contiguous() for st in rung_st}
    log(f"loaded {len(rung_st)} rung stories, {len(cell.vox)} voxels t={time.time()-t0:.0f}s")
    cell_out = {"6": {}, "20": {}}; calib = {}; Pfull = precision(cell.Sigma)
    def run_block(ev, story, W, P, name, Tcal):
        o = S.score_block(ev, story, cell, W, P, cond, seed_of(a.subject, cond, rung, ev, "etc"))
        for k in ("6", "20"):
            T_, th, tht = Tcal[k]
            cell_out[k][ev] = {**summarise(o[k], S.tx[k]["foil_valid"][o[k]["sel"]], None, T_, th), "th": th, "thtn": tht, "T": T_, "model": name}
    oc = S.score_block(CALIB, CALIB, cell, cell.W, Pfull, cond, seed_of(a.subject, cond, rung, "calib", "etc"))
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
    if len(rung_st) > 1:
        for i, (h_, Wh, Ph) in enumerate(fold_models(cell, stim, resp_sel, rung_st)):
            run_block(h_, h_, Wh, Ph, h_, Tcal)
            if i % 10 == 0: log(f"fold {i}/{len(rung_st)} t={time.time()-t0:.0f}s")
    np.savez_compressed(f"{a.out}/{a.subject}_{a.cell}.npz.tmp.npz", **flatten(cell_out)); os.replace(f"{a.out}/{a.subject}_{a.cell}.npz.tmp.npz", f"{a.out}/{a.subject}_{a.cell}.npz")
    hours = sum(stim[s].shape[0] for s in rung_st) * 2 / 3600
    res = {a.cell: {"cond": cond, "rung": rung, "dB": pseudo_db(cond), "ultrasound": is_us(cond), "rung_stories": rung_st, "hours": hours, "n_train_stories": len(rung_st),
                    "q": None, "lam": None, "trials": [{"alpha_grid": 10, "nboots": 50}], "calib": calib, "n_params_trainable": int(cell.W.numel()), "n_params_pca": 0,
                    "n_models": 1 + (len(rung_st) if len(rung_st) > 1 else 0), "loso": len(rung_st) > 1, "n_voxels": int(len(cell.vox)),
                    "gpu_s": {"etc_total": time.time() - t0}, "wall_s": time.time() - t0, "readout": "encode-then-compare (timing-matched pool and cohort, K=600)",
                    "kappa": S.kappa, "pool_K": S.K, "pool_seed": POOL_SEED}}
    cj = f"{a.out}/{a.subject}_{a.cell}_cell.json"; json.dump(res, open(cj + ".tmp", "w"), indent=1); os.replace(cj + ".tmp", cj)
    log(f"done {a.cell} t={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
