"""Subject-independent text side for the contrastive verifier.

Timing (Tang's conventions, verified in prep_resp.py): response row i of a story <-> TR acquired over
stimulus time [2i+10, 2i+12) s. Anchor a: window ends at T_a = 2a+12 s; brain input = rows a+1..a+4.
  * 6 s windows  : words with time in (T_a-6, T_a]   (<=32 GTR tokens; longer windows are left-truncated)
  * 20 s segments: words in (T_a-20, T_a], a = 9, 14, 19, ... ; scored as 10 sub-anchors b=a-9..a, each a
                   6 s sub-window (T_b-6, T_b] of the candidate's words (no context outside the segment)
Foils (decoder-free, Phase A), all with the TRUE segment's word count and word times:
  * NN-r : n consecutive words from another story's window that is the r-th nearest (GTR cosine) to the truth,
           r in RANKS, plus 3 random windows  -> continuous distance axis
  * edits: negation flip, one pronoun swap, all pronoun swaps, k content-word substitutions (same POS tag,
           in-vocabulary replacement words)
Distance of a foil = 1 - cos(GTR(true text), GTR(foil text)), full text, no truncation.
<unk> control: fraction of words outside Tang's GPT-1 word vocabulary, for truth and every foil.
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, json, time, numpy as np, torch
REPO = VP.TANG_CODE; sys.path.insert(0, REPO)
from utils_ridge.textgrid import TextGrid
from utils_ridge.stimulus_utils import load_simulated_trfiles
from utils_ridge.dsutils import make_word_ds
from LanguageModel import STOPWORDS
import nltk
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

OUT = f"{VP.WORK}/text"; os.makedirs(OUT, exist_ok=True)
SESS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 20]
CALIB, TEST = "fromboyhoodtofatherhood", "wheretheressmoke"
RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
NRAND = 3
EDITS6 = ["neg1", "pro1", "proall", "sub1", "sub2", "sub4"]
EDITS20 = ["neg1", "pro1", "proall", "sub1", "sub2", "sub4", "sub8"]
NEG = {"is": "isn't", "was": "wasn't", "are": "aren't", "were": "weren't", "do": "don't", "does": "doesn't", "did": "didn't",
       "can": "can't", "could": "couldn't", "would": "wouldn't", "will": "won't", "should": "shouldn't", "have": "haven't",
       "has": "hasn't", "had": "hadn't", "never": "always", "always": "never", "nothing": "everything", "everything": "nothing",
       "nobody": "everybody", "everybody": "nobody", "not": "also"}
NEG.update({v: k for k, v in list(NEG.items()) if v not in NEG and k not in ("not",)})
PRO = {"he": "she", "she": "he", "him": "her", "his": "her", "her": "him", "himself": "herself", "herself": "himself",
       "i": "you", "you": "i", "me": "you", "my": "your", "your": "my", "mine": "yours", "yours": "mine", "myself": "yourself",
       "yourself": "myself", "we": "they", "they": "we", "us": "them", "them": "us", "our": "their", "their": "our"}
CTAGS = {"NN", "NNS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "JJ"}
DEV = "cuda"

def load_words():
    s2s = json.load(open(f"{VP.TANG_BOX}/data_train/sess_to_story.json"))
    stories = [st for s in SESS for st in s2s[str(s)]] + [CALIB, TEST]
    rd = json.load(open(f"{VP.DS}/respdict.json"))
    grids = {st: TextGrid(open(f"{VP.DS}/TextGrids/{st}.TextGrid").read()) for st in stories}
    ws = make_word_ds(grids, load_simulated_trfiles({st: rd[st] for st in stories}))
    W = {st: (list(ws[st].data), np.asarray(ws[st].data_times, float), rd[st] - 20) for st in stories}
    return stories, W

class GTR:
    def __init__(self):
        self.st = SentenceTransformer("sentence-transformers/gtr-t5-base", device=DEV).half()
        self.tok = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")
        self.enc = self.st[0].auto_model; self.dense = self.st[2]
        self.cache = {}
    @torch.no_grad()
    def embed(self, texts, max_tok=32, both=False, bs=1024):
        """returns (st_normalised, v2t_raw_meanpool) ; left-truncates to the LAST max_tok tokens (EOS kept)"""
        self.cache = {}; key = lambda t: (t, max_tok)   # per-call cache (dedup only; keeps RAM bounded)
        todo = list(dict.fromkeys(texts))
        order = sorted(range(len(todo)), key=lambda i: len(todo[i]))
        for i in range(0, len(order), bs):
            batch = [todo[j] for j in order[i:i + bs]]
            ids = [self.tok(t).input_ids for t in batch]
            if max_tok: ids = [x[-max_tok:] for x in ids]
            L = max(len(x) for x in ids)
            inp = torch.zeros(len(ids), L, dtype=torch.long); att = torch.zeros(len(ids), L, dtype=torch.long)
            for k, x in enumerate(ids): inp[k, :len(x)] = torch.tensor(x); att[k, :len(x)] = 1
            inp, att = inp.to(DEV), att.to(DEV)
            h = self.enc(input_ids=inp, attention_mask=att).last_hidden_state.float()
            m = att.unsqueeze(-1).float(); v2t = (h * m).sum(1) / m.sum(1)
            st = torch.nn.functional.normalize(self.dense({"sentence_embedding": v2t.half()})["sentence_embedding"].float(), dim=-1)
            for k, t in enumerate(batch): self.cache[key(t)] = (st[k].cpu().numpy().astype(np.float16), v2t[k].cpu().numpy().astype(np.float16))
        st = np.stack([self.cache[key(t)][0] for t in texts]); v2 = np.stack([self.cache[key(t)][1] for t in texts])
        return (st, v2) if both else st
    def ntok(self, t): return len(self.tok(t).input_ids)

def main():
    t0 = time.time(); rng = np.random.default_rng(1234)
    stories, W = load_words(); sid = {st: i for i, st in enumerate(stories)}
    gpt_vocab = set(json.load(open(f"{VP.TANG_BOX}/data_lm/perceived/vocab.json")))
    dec_vocab = set(json.load(open(f"{VP.TANG_BOX}/data_lm/decoder_vocab.json")))
    oov = lambda ws: float(np.mean([w not in gpt_vocab for w in ws])) if len(ws) else 0.0
    # POS tags per story (in context) and replacement pools (training + calib stories only)
    tags = {st: [t for _, t in nltk.pos_tag(W[st][0])] for st in stories}
    from collections import Counter
    cnt = Counter(); pool = {t: set() for t in CTAGS}
    for st in stories[:-1]:
        for w, t in zip(W[st][0], tags[st]): cnt[(w, t)] += 1
    for (w, t), c in cnt.items():
        if t in CTAGS and c >= 2 and w not in STOPWORDS and w in gpt_vocab and w in dec_vocab and w.isalpha(): pool[t].add(w)
    pool = {t: sorted(v) for t, v in pool.items()}
    gtr = GTR()
    # ---- anchors
    A = {}
    for st in stories:
        words, times, n = W[st]
        a = np.arange(0, n - 4); T = 2 * a + 12.0
        idx6 = [np.where((times > T_ - 6) & (times <= T_))[0] for T_ in T]
        a20 = np.arange(9, n - 4, 5)
        idx20 = [np.where((times > 2 * x + 12.0 - 20) & (times <= 2 * x + 12.0))[0] for x in a20]
        A[st] = dict(a=a, T=T, idx6=idx6, a20=a20, idx20=idx20)
    text = lambda st, ii: " ".join(W[st][0][i] for i in ii)
    # ---- NN pools: 6 s windows and 20 s windows at all anchors with >=1 word
    P6 = [(st, k) for st in stories for k in range(len(A[st]["a"])) if len(A[st]["idx6"][k]) >= 1]
    P6_txt = [text(st, A[st]["idx6"][k]) for st, k in P6]
    P6_emb = torch.tensor(gtr.embed(P6_txt, max_tok=32)).float().to(DEV)
    P6_sid = torch.tensor([sid[st] for st, _ in P6]).to(DEV)
    idx20_all = {st: [np.where((W[st][1] > T_ - 20) & (W[st][1] <= T_))[0] for T_ in A[st]["T"]] for st in stories}
    P20 = [(st, k) for st in stories for k in range(len(A[st]["a"])) if len(idx20_all[st][k]) >= 1]
    P20_emb = torch.tensor(gtr.embed([text(st, idx20_all[st][k]) for st, k in P20], max_tok=None)).float().to(DEV)
    P20_sid = torch.tensor([sid[st] for st, _ in P20]).to(DEV)
    print(f"pools 6s {len(P6)} 20s {len(P20)} t={time.time()-t0:.0f}s", flush=True)

    def make_foils(st, ii, kind):
        """ii: word indices of the true segment. returns list of (name, words or None)"""
        words, times, _ = W[st]; n = len(ii); true_w = [words[i] for i in ii]; tg = [tags[st][i] for i in ii]
        e = torch.tensor(gtr.embed([" ".join(true_w)], max_tok=32 if kind == 6 else None)).float().to(DEV)
        Pe, Ps, P = (P6_emb, P6_sid, P6) if kind == 6 else (P20_emb, P20_sid, P20)
        sim = (Pe @ e[0]); sim[Ps == sid[st]] = -9
        order = torch.argsort(sim, descending=True).cpu().numpy()
        n_other = int((Ps != sid[st]).sum())
        picks = [order[r - 1] for r in RANKS] + list(order[rng.choice(n_other, NRAND, replace=False)])
        out = []
        for j, r in zip(picks, [f"nn{r}" for r in RANKS] + [f"rand{i}" for i in range(NRAND)]):
            s2, k2 = P[j]; src = (A[s2]["idx6"][k2] if kind == 6 else idx20_all[s2][k2])
            j0 = int(src[0]); L = len(W[s2][0]); j0 = min(j0, L - n)
            out.append((r, W[s2][0][j0:j0 + n]))
        for ed in (EDITS6 if kind == 6 else EDITS20):
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

    # ---- 6 s windows: evaluated anchors (>=3 words)
    rec6 = dict(story=[], anchor=[], nw=[], ntok=[], true_oov=[], e_st=[], e_v2t=[], foil_e=[], foil_d=[], foil_valid=[], foil_oov=[])
    foil_names6 = None; ex6 = {}
    for si, st in enumerate(stories):
        Ast = A[st]
        for k, ii in enumerate(Ast["idx6"]):
            if len(ii) < 3: continue
            true_w, foils = make_foils(st, ii, 6)
            if foil_names6 is None: foil_names6 = [f[0] for f in foils]
            ttxt = " ".join(true_w); ftxt = [" ".join(f[1]) if f[1] is not None else ttxt for f in foils]
            est, ev2 = gtr.embed([ttxt], both=True)
            fe = gtr.embed(ftxt)
            d = 1 - (fe.astype(np.float32) @ est[0].astype(np.float32))
            rec6["story"].append(si); rec6["anchor"].append(int(Ast["a"][k])); rec6["nw"].append(len(ii)); rec6["ntok"].append(gtr.ntok(ttxt))
            rec6["true_oov"].append(oov(true_w)); rec6["e_st"].append(est[0]); rec6["e_v2t"].append(ev2[0]); rec6["foil_e"].append(fe)
            rec6["foil_d"].append(d); rec6["foil_valid"].append([f[1] is not None for f in foils]); rec6["foil_oov"].append([oov(f[1]) if f[1] else np.nan for f in foils])
            if st == TEST and k % 25 == 0: ex6[int(Ast["a"][k])] = {"true": ttxt, **{f[0]: (" ".join(f[1]) if f[1] else None) for f in foils}}
        print(f"6s {st} {len(rec6['story'])} t={time.time()-t0:.0f}s", flush=True)
    np.savez(f"{OUT}/win6.npz", **{k: np.asarray(v) for k, v in rec6.items()}, foil_names=np.array(foil_names6), stories=np.array(stories))
    json.dump(ex6, open(f"{OUT}/examples6_test.json", "w"), indent=1)
    # ---- 20 s segments (all anchors step 5), candidates = true + foils, each as 10 sub-window embeddings
    N20 = sum(len(A[st]["a20"]) for st in stories); K20 = len(RANKS) + NRAND + len(EDITS20)
    sub = np.lib.format.open_memmap(f"{OUT}/seg20_sub.npy", mode="w+", dtype=np.float16, shape=(N20, 1 + K20, 10, 768))
    rec20 = dict(story=[], anchor=[], nw=[], true_oov=[], sub_mask=[], sub_nw=[], foil_d=[], foil_valid=[], foil_oov=[], e_full=[])
    foil_names20 = None; ex20 = {}; r = 0
    for si, st in enumerate(stories):
        Ast = A[st]; times = W[st][1]
        for k, ii in enumerate(Ast["idx20"]):
            a = int(Ast["a20"][k]); t = times[ii]
            true_w, foils = make_foils(st, ii, 20) if len(ii) >= 1 else ([], [(f, None) for f in [f"nn{x}" for x in RANKS] + [f"rand{i}" for i in range(NRAND)] + EDITS20])
            if foil_names20 is None and len(ii) >= 1: foil_names20 = [f[0] for f in foils]
            bins = [np.where((t > 2 * b + 12.0 - 6) & (t <= 2 * b + 12.0))[0] for b in range(a - 9, a + 1)]
            cands = [true_w] + [f[1] if f[1] is not None else true_w for f in foils]
            txts = [[" ".join(c[j] for j in bb) for bb in bins] for c in cands]
            flat = gtr.embed([x for row in txts for x in row])
            sub[r] = flat.reshape(1 + K20, 10, 768)
            full = gtr.embed([" ".join(c) for c in cands], max_tok=None)
            d = 1 - (full[1:].astype(np.float32) @ full[0].astype(np.float32))
            rec20["story"].append(si); rec20["anchor"].append(a); rec20["nw"].append(len(ii)); rec20["true_oov"].append(oov(true_w))
            rec20["sub_mask"].append([len(bb) > 0 for bb in bins]); rec20["sub_nw"].append([len(bb) for bb in bins])
            rec20["foil_d"].append(d); rec20["foil_valid"].append([f[1] is not None for f in foils]); rec20["foil_oov"].append([oov(f[1]) if f[1] else np.nan for f in foils])
            rec20["e_full"].append(full[0])
            if st == TEST and k % 5 == 0: ex20[a] = {"true": " ".join(true_w), **{f[0]: (" ".join(f[1]) if f[1] else None) for f in foils}}
            r += 1
        print(f"20s {st} {r} t={time.time()-t0:.0f}s", flush=True)
    sub.flush()
    np.savez(f"{OUT}/seg20.npz", **{k: np.asarray(v) for k, v in rec20.items()}, foil_names=np.array(foil_names20), stories=np.array(stories))
    json.dump(ex20, open(f"{OUT}/examples20_test.json", "w"), indent=1)
    # ---- training targets: 6 s window embeddings at EVERY anchor (>=1 word), all stories
    tr = dict(story=[], anchor=[], nw=[], e=[])
    for si, st in enumerate(stories):
        for k, ii in enumerate(A[st]["idx6"]):
            if len(ii) < 1: continue
            tr["story"].append(si); tr["anchor"].append(int(A[st]["a"][k])); tr["nw"].append(len(ii))
        es = gtr.embed([text(st, ii) for ii in A[st]["idx6"] if len(ii) >= 1]); tr["e"].extend(list(es))
    np.savez(f"{OUT}/train6.npz", **{k: np.asarray(v) for k, v in tr.items()}, stories=np.array(stories))
    meta = {"stories": stories, "n_rows": {st: int(W[st][2]) for st in stories}, "RANKS": RANKS, "EDITS6": EDITS6, "EDITS20": EDITS20,
            "foil_names6": foil_names6, "foil_names20": foil_names20, "pool_sizes": {t: len(v) for t, v in pool.items()},
            "n_win6": len(rec6["story"]), "n_seg20": N20, "frac_win6_over_32tok": float(np.mean(np.array(rec6["ntok"]) > 32)),
            "median_words_6s": float(np.median(rec6["nw"])), "prep_text_s": time.time() - t0}
    json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
    print(json.dumps(meta)[:2000])

if __name__ == "__main__":
    main()
