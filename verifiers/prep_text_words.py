"""[Phase B copy: identical draws; additionally saves the foil WORD lists to text_words.json.gz for encode-then-compare]
Batched re-implementation of prep_text.main (same definitions, same output files; different random draws).
All GTR calls are large batches; nearest-neighbour search is a batched matmul + topk."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, gzip, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prep_text import (load_words, GTR, OUT, CALIB, TEST, RANKS, NRAND, EDITS6, EDITS20, NEG, PRO, CTAGS, DEV)
from LanguageModel import STOPWORDS
import nltk
from collections import Counter

def main():
    t0 = time.time(); rng = np.random.default_rng(1234)
    stories, W = load_words(); sid = {st: i for i, st in enumerate(stories)}
    gpt_vocab = set(json.load(open(f"{VP.TANG_BOX}/data_lm/perceived/vocab.json")))
    dec_vocab = set(json.load(open(f"{VP.TANG_BOX}/data_lm/decoder_vocab.json")))
    oov = lambda ws: float(np.mean([w not in gpt_vocab for w in ws])) if len(ws) else 0.0
    tags = {st: [t for _, t in nltk.pos_tag(W[st][0])] for st in stories}
    cnt = Counter(); pool = {t: set() for t in CTAGS}
    for st in stories[:-1]:
        for w, t in zip(W[st][0], tags[st]): cnt[(w, t)] += 1
    for (w, t), c in cnt.items():
        if t in CTAGS and c >= 2 and w not in STOPWORDS and w in gpt_vocab and w in dec_vocab and w.isalpha(): pool[t].add(w)
    pool = {t: sorted(v) for t, v in pool.items()}
    gtr = GTR()
    A = {}
    for st in stories:
        words, times, n = W[st]
        a = np.arange(0, n - 4); T = 2 * a + 12.0
        A[st] = dict(a=a, T=T, idx6=[np.where((times > x - 6) & (times <= x))[0] for x in T],
                     idx20all=[np.where((times > x - 20) & (times <= x))[0] for x in T],
                     a20=np.arange(9, n - 4, 5))
        A[st]["idx20"] = [A[st]["idx20all"][x] for x in A[st]["a20"]]
    text = lambda st, ii: " ".join(W[st][0][i] for i in ii)
    P6 = [(st, k) for st in stories for k in range(len(A[st]["a"])) if len(A[st]["idx6"][k]) >= 1]
    P6_emb = torch.tensor(gtr.embed([text(st, A[st]["idx6"][k]) for st, k in P6], max_tok=32)).float().to(DEV)
    P6_sid = torch.tensor([sid[st] for st, _ in P6]).to(DEV)
    P20 = [(st, k) for st in stories for k in range(len(A[st]["a"])) if len(A[st]["idx20all"][k]) >= 1]
    P20_emb = torch.tensor(gtr.embed([text(st, A[st]["idx20all"][k]) for st, k in P20], max_tok=None, bs=256)).float().to(DEV)
    P20_sid = torch.tensor([sid[st] for st, _ in P20]).to(DEV)
    print(f"pools 6s {len(P6)} 20s {len(P20)} t={time.time()-t0:.0f}s", flush=True)

    def nn_picks(E_true, story_ids, Pe, Ps, P):
        picks = []
        for i in range(0, len(E_true), 256):
            sim = E_true[i:i + 256] @ Pe.T
            sim[Ps[None] == story_ids[i:i + 256, None]] = -9
            top = torch.topk(sim, max(RANKS), dim=1).indices.cpu().numpy()
            for r_, row in enumerate(top):
                other = np.where((Ps != story_ids[i + r_]).cpu().numpy())[0]
                picks.append([row[r - 1] for r in RANKS] + list(rng.choice(other, NRAND, replace=False)))
        return picks

    def foils_from(st, ii, picks, P, idx_key, edits):
        words = W[st][0]; n = len(ii); true_w = [words[i] for i in ii]; tg = [tags[st][i] for i in ii]; out = []
        for j, name in zip(picks, [f"nn{r}" for r in RANKS] + [f"rand{i}" for i in range(NRAND)]):
            s2, k2 = P[j]; src = A[s2][idx_key][k2]; L = len(W[s2][0]); j0 = min(int(src[0]), L - n)
            out.append((name, W[s2][0][j0:j0 + n]))
        for ed in edits:
            w2 = list(true_w)
            if ed == "neg1":
                el = [i for i, w in enumerate(w2) if w in NEG]
                if not el: out.append((ed, None)); continue
                i = el[rng.integers(len(el))]; w2[i] = NEG[w2[i]]
            elif ed in ("pro1", "proall"):
                el = [i for i, w in enumerate(w2) if w in PRO]
                if not el: out.append((ed, None)); continue
                for i in (el if ed == "proall" else [el[rng.integers(len(el))]]): w2[i] = PRO[true_w[i]]
            else:
                k = int(ed[3:]); el = [i for i, (w, t) in enumerate(zip(w2, tg)) if t in CTAGS and w not in STOPWORDS and len(pool[t]) > 1]
                if len(el) < k: out.append((ed, None)); continue
                for i in rng.choice(el, k, replace=False):
                    c = true_w[i]
                    while c == true_w[i]: c = pool[tg[i]][rng.integers(len(pool[tg[i]]))]
                    w2[i] = c
            out.append((ed, w2))
        return true_w, out

    # ---------------- 6 s windows
    ev = [(st, k) for st in stories for k, ii in enumerate(A[st]["idx6"]) if len(ii) >= 3]
    ttxt = [text(st, A[st]["idx6"][k]) for st, k in ev]
    est, ev2 = gtr.embed(ttxt, both=True)
    picks = nn_picks(torch.tensor(est).float().to(DEV), torch.tensor([sid[st] for st, _ in ev]).to(DEV), P6_emb, P6_sid, P6)
    F6 = [foils_from(st, A[st]["idx6"][k], pk, P6, "idx6", EDITS6) for (st, k), pk in zip(ev, picks)]
    names6 = [f[0] for f in F6[0][1]]
    WORDS = {"w6": [[st, int(A[st]["a"][k]), [int(i) for i in A[st]["idx6"][k]], [f[1] for f in fl]] for (st, k), (tw, fl) in zip(ev, F6)], "w20": []}
    ftxt = [" ".join(f[1]) if f[1] is not None else tt for (tw, fl), tt in zip(F6, ttxt) for f in fl]
    fe = gtr.embed(ftxt).reshape(len(ev), len(names6), 768)
    d6 = 1 - np.einsum("nkd,nd->nk", fe.astype(np.float32), est.astype(np.float32))
    np.savez(f"{OUT}/win6.npz", story=np.array([sid[st] for st, _ in ev]), anchor=np.array([int(A[st]["a"][k]) for st, k in ev]),
             nw=np.array([len(A[st]["idx6"][k]) for st, k in ev]), ntok=np.array([gtr.ntok(t) for t in ttxt]),
             true_oov=np.array([oov(tw) for tw, _ in F6]), e_st=est, e_v2t=ev2, foil_e=fe, foil_d=d6,
             foil_valid=np.array([[f[1] is not None for f in fl] for _, fl in F6]),
             foil_oov=np.array([[oov(f[1]) if f[1] else np.nan for f in fl] for _, fl in F6]),
             foil_names=np.array(names6), stories=np.array(stories))
    ex6 = {int(A[st]["a"][k]): {"true": tt, **{f[0]: (" ".join(f[1]) if f[1] else None) for f in fl}}
           for (st, k), tt, (_, fl) in zip(ev, ttxt, F6) if st == TEST and k % 25 == 0}
    json.dump(ex6, open(f"{OUT}/examples6_test.json", "w"), indent=1)
    ntok6 = np.array([gtr.ntok(t) for t in ttxt])
    print(f"6s done n={len(ev)} t={time.time()-t0:.0f}s", flush=True)
    del fe
    # ---------------- 20 s segments
    seg = [(st, k) for st in stories for k in range(len(A[st]["a20"]))]
    K20 = len(RANKS) + NRAND + len(EDITS20)
    sub = np.lib.format.open_memmap(f"{OUT}/seg20_sub.npy", mode="w+", dtype=np.float16, shape=(len(seg), 1 + K20, 10, 768))
    rec = dict(story=[], anchor=[], nw=[], true_oov=[], sub_mask=[], sub_nw=[], foil_d=[], foil_valid=[], foil_oov=[], e_full=[])
    ex20 = {}; names20 = [f"nn{r}" for r in RANKS] + [f"rand{i}" for i in range(NRAND)] + EDITS20
    CH = 400
    for c0 in range(0, len(seg), CH):
        part = seg[c0:c0 + CH]
        ttx = [text(st, A[st]["idx20"][k]) for st, k in part]
        etrue = gtr.embed(ttx, max_tok=None, bs=256)
        has = [len(A[st]["idx20"][k]) >= 1 for st, k in part]
        hi = [i for i, h in enumerate(has) if h]
        pk = nn_picks(torch.tensor(etrue[hi]).float().to(DEV), torch.tensor([sid[part[i][0]] for i in hi]).to(DEV), P20_emb, P20_sid, P20)
        pkd = dict(zip(hi, pk))
        cands_all, bins_all = [], []
        for i, (st, k) in enumerate(part):
            ii = A[st]["idx20"][k]; a = int(A[st]["a20"][k]); t = W[st][1][ii]
            if has[i]: tw, fl = foils_from(st, ii, pkd[i], P20, "idx20all", EDITS20)
            else: tw, fl = [], [(nm, None) for nm in names20]
            bins = [np.where((t > 2 * b + 12.0 - 6) & (t <= 2 * b + 12.0))[0] for b in range(a - 9, a + 1)]
            cands = [tw] + [f[1] if f[1] is not None else tw for f in fl]
            cands_all.append((tw, fl, cands)); bins_all.append(bins)
            WORDS["w20"].append([st, a, [int(i) for i in ii], [f[1] for f in fl]])
            rec["story"].append(sid[st]); rec["anchor"].append(a); rec["nw"].append(len(ii)); rec["true_oov"].append(oov(tw))
            rec["sub_mask"].append([len(bb) > 0 for bb in bins]); rec["sub_nw"].append([len(bb) for bb in bins])
            rec["foil_valid"].append([f[1] is not None for f in fl]); rec["foil_oov"].append([oov(f[1]) if f[1] else np.nan for f in fl])
            if st == TEST and k % 5 == 0: ex20[a] = {"true": " ".join(tw), **{f[0]: (" ".join(f[1]) if f[1] else None) for f in fl}}
        flat = [" ".join(c[j] for j in bb) for (_, _, cands), bins in zip(cands_all, bins_all) for c in cands for bb in bins]
        E = gtr.embed(flat).reshape(len(part), 1 + K20, 10, 768); sub[c0:c0 + len(part)] = E
        full = gtr.embed([" ".join(c) for _, _, cands in cands_all for c in cands], max_tok=None, bs=256).reshape(len(part), 1 + K20, 768).astype(np.float32)
        rec["foil_d"].extend(list(1 - np.einsum("nkd,nd->nk", full[:, 1:], full[:, 0]))); rec["e_full"].extend(list(full[:, 0].astype(np.float16)))
        print(f"20s {c0+len(part)}/{len(seg)} t={time.time()-t0:.0f}s", flush=True)
    sub.flush()
    with gzip.open(f"{OUT}/text_words.json.gz", "wt") as fz: json.dump(WORDS, fz)
    np.savez(f"{OUT}/seg20.npz", **{k: np.asarray(v) for k, v in rec.items()}, foil_names=np.array(names20), stories=np.array(stories))
    json.dump(ex20, open(f"{OUT}/examples20_test.json", "w"), indent=1)
    # ---------------- training targets (all anchors with >=1 word)
    tr = [(st, k) for st in stories for k in range(len(A[st]["a"])) if len(A[st]["idx6"][k]) >= 1]
    e = gtr.embed([text(st, A[st]["idx6"][k]) for st, k in tr])
    np.savez(f"{OUT}/train6.npz", story=np.array([sid[st] for st, _ in tr]), anchor=np.array([int(A[st]["a"][k]) for st, k in tr]),
             nw=np.array([len(A[st]["idx6"][k]) for st, k in tr]), e=e, stories=np.array(stories))
    meta = {"stories": stories, "n_rows": {st: int(W[st][2]) for st in stories}, "RANKS": RANKS, "EDITS6": EDITS6, "EDITS20": EDITS20,
            "foil_names6": names6, "foil_names20": names20, "pool_sizes": {t: len(v) for t, v in pool.items()},
            "n_win6": len(ev), "n_seg20": len(seg), "frac_win6_over_32tok": float(np.mean(ntok6 > 32)),
            "median_words_6s": float(np.median([len(A[st]["idx6"][k]) for st, k in ev])), "prep_text_s": time.time() - t0, "impl": "prep_text2 (batched)"}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta)[:1500])

if __name__ == "__main__":
    main()
