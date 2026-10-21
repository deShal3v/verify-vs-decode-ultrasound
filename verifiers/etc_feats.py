"""Phase B, encode-then-compare (E-t-C) verifier, step 1 (subject-independent): TR-level GPT-1 features of every
candidate text (truth + foils) of every Phase A window, in Tang's stimulus space before z-scoring.

Candidate c of window (story s, anchor a): words w_1..w_n at times t_1..t_n (foils carry the true segment's times).
Word features e_i = Tang's LMFeatures.make_stim(candidate words) (GPT-1 'perceived', layer 9, 5-word context,
context only inside the candidate = 'no context'). TR features on the window's TR span (trimmed TR index j, TR time
2j+11 s as in Tang's DataSequence): F_c[j] = sum_i lanczos(2j+11 - t_i) e_i (Tang's lanczosfun, window 3, 0.5 Hz);
S_c[j] = sum_i lanczos(2j+11 - t_i).
6 s windows (end T_a = 2a+12): rows a-1..a+4, TR span j = a-5..a+3 (9 TRs). 20 s: rows a-10..a+4, j = a-14..a+3 (18).
Outputs $WORK_DIR/etc/{F6.npy (N6,K6+1,9,768) f16, S6.npy (N6,9) f32, F20.npy, S20.npy, meta.json}; slot 0 = truth,
slots 1.. = foils in Phase A order (invalid foils: copy of truth, masked by foil_valid)."""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, gzip, time, numpy as np, torch
REPO = VP.TANG_CODE; sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.backends.cuda.matmul.allow_tf32 = False
import config
from GPT import GPT
from utils_ridge.interpdata import lanczosfun
from prep_text import load_words
OUT = f"{VP.WORK}/etc"; os.makedirs(OUT, exist_ok=True)

def make_stim_batch(gpt, cands, ctx=config.GPT_WORDS, bs=4096):
    """LMFeatures.make_stim for many word lists at once (identical per-list result): returns list of (n_i, 768)"""
    rows, owner = [], []
    for ci, words in enumerate(cands):
        ids = gpt.encode(words); n = len(ids); nctx = ctx + 1
        arr = np.zeros([n, nctx], dtype=np.int64) + gpt.UNK_ID
        for i in range(n):
            seg = ids[i:i + nctx]; arr[i, :len(seg)] = seg
        rows.append(arr); owner.append(n)
    A = torch.from_numpy(np.concatenate(rows))
    H = np.empty((A.shape[0], ctx + 1, 768), np.float32)
    with torch.no_grad():
        for i in range(0, A.shape[0], bs):
            ids = A[i:i + bs].to("cuda")
            H[i:i + bs] = gpt.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[config.GPT_LAYER].float().cpu().numpy()
    out, r = [], 0
    for n in owner:
        h = H[r:r + n]; r += n
        # make_stim: first ctx words from row 0 positions 0..ctx-1, the rest from position ctx of rows 0..n-ctx-1
        out.append(np.vstack([h[0, :ctx], h[:max(n - ctx, 0), ctx]])[:n])
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--kinds", default="6,20"); a = ap.parse_args()
    t0 = time.time()
    stories, W = load_words()
    WD = json.load(gzip.open(f"{VP.WORK}/text_b/text_words.json.gz", "rt"))
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=vocab, device="cuda")
    # only layer 9 is needed: drop blocks 10-12 (GPT-1 is post-LN, so hidden_states[9] is unchanged); checked on a sample
    ids = torch.from_numpy(np.random.default_rng(0).integers(0, len(vocab), (64, 6))).to("cuda")
    with torch.no_grad(): ref = gpt.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[config.GPT_LAYER]
    gpt.model.transformer.h = gpt.model.transformer.h[:config.GPT_LAYER]
    with torch.no_grad(): new = gpt.model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True).hidden_states[config.GPT_LAYER]
    assert torch.allclose(ref, new), float((ref - new).abs().max())
    meta = {"shard": a.shard, "nshards": a.nshards}
    for kind, span0, span1 in [x for x in (("6", -5, 3), ("20", -14, 3)) if x[0] in a.kinds.split(",")]:
        ent_all = WD["w" + kind]; Nall = len(ent_all); K1 = 1 + len(ent_all[0][3]); L = span1 - span0 + 1
        lo_i = Nall * a.shard // a.nshards; hi_i = Nall * (a.shard + 1) // a.nshards; ent = ent_all[lo_i:hi_i]; N = len(ent)
        sfx = "" if a.nshards == 1 else f"_shard{a.shard}"
        F = np.lib.format.open_memmap(f"{OUT}/F{kind}{sfx}.npy", mode="w+", dtype=np.float16, shape=(N, K1, L, 768))
        S = np.zeros((N, L), np.float32); maxabs = 0.0
        CH = 2000 if kind == "6" else 400
        for c0 in range(0, N, CH):
            part = ent[c0:c0 + CH]; cands, meta_c = [], []
            for st, anc, ii, foils in part:
                words = [W[st][0][i] for i in ii]; times = W[st][1][ii]
                for k, fw in enumerate([words] + foils):
                    cands.append(fw if fw is not None else words); meta_c.append(times)
            feats = make_stim_batch(gpt, cands)
            q = 0
            for r, (st, anc, ii, foils) in enumerate(part):
                times = W[st][1][ii]; trt = 2.0 * (anc + np.arange(span0, span1 + 1)) + 11.0
                Lm = np.stack([lanczosfun(0.5, x - times, 3) for x in trt]) if len(times) else np.zeros((L, 0))
                S[c0 + r] = Lm.sum(1)
                for k in range(K1):
                    e = feats[q]; q += 1
                    v = Lm @ e if len(e) else np.zeros((L, 768), np.float32)
                    maxabs = max(maxabs, float(np.abs(v).max())); F[c0 + r, k] = v.astype(np.float16)
            print(f"{kind}s {c0 + len(part)}/{N} t={time.time()-t0:.0f}s", flush=True)
        F.flush(); np.save(f"{OUT}/S{kind}{sfx}.npy", S)
        meta[kind] = {"N": N, "range": [lo_i, hi_i], "K1": K1, "span": [span0, span1], "maxabs_F": maxabs}
    meta["etc_feats_s"] = time.time() - t0
    ok = all(os.path.exists(f"{OUT}/F{k}{'' if a.nshards == 1 else f'_shard{a.shard}'}.npy") and os.path.exists(f"{OUT}/S{k}{'' if a.nshards == 1 else f'_shard{a.shard}'}.npy") for k in ("6", "20"))
    if ok: json.dump(meta, open(f"{OUT}/meta{'' if a.nshards == 1 else f'_shard{a.shard}'}.json", "w"), indent=1)
    print(json.dumps(meta), "complete" if ok else "partial")

if __name__ == "__main__":
    main()
