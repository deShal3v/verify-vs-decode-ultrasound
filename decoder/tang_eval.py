"""Evaluate Tang decoder predictions with the repository's metrics (decoding/utils_eval.py), with
the known bugs fixed:
  * datasets.load_metric (removed in datasets>=3) -> evaluate.load (same HF 'bleu'/'meteor' metrics);
  * WER (issue #9): jiwer gets the window as ONE string, not a list of words; report error rate;
  * story score (issue #10): story score = mean of window scores (as in the paper's "story similarity");
  * nulls: Tang's paper uses 200 LM-only sequences (the script defaults to 10). We generate them with a
    batched re-implementation of utils_eval.generate_null (same algorithm: beam 2*EXTENSIONS, LM nucleus
    proposals, random likelihoods, same predicted word times); the GPT forward is batched across nulls
    and only the last position's softmax is computed (identical numbers for equal-length contexts).
    --check_seq runs the original sequential generate_null for a few nulls to compare distributions.
  * load_transcript, windows and segment_data are imported from Tang et al.'s utils_eval.py (not copied).
Output: scores/<subject>/<experiment>/<task>_eval.npz + .json (window scores, null window scores, z).
"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, time, json, argparse
from functools import lru_cache
REPO = VP.TANG_CODE
sys.path.insert(0, REPO)
import numpy as np, torch
import config
import LanguageModel as LMmod
LMmod.stemmer.stem = lru_cache(maxsize=None)(LMmod.stemmer.stem)
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel, get_nucleus, context_filter, INIT
from jiwer import wer
from bert_score import BERTScorer
import evaluate
from utils_ridge.textgrid import TextGrid

# transcript loading and 20-s windowing: Tang et al.'s own functions (decoding/utils_eval.py at the pinned commit)
import utils_eval
from utils_eval import load_transcript, windows, segment_data

def load_lm(gpt_checkpoint):
    with open(os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "vocab.json")) as f: gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json")) as f: decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, gpt_checkpoint, "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    return gpt, LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)

def generate_null_sequential(pred_times, gpt, lm, n, gpt_checkpoint="perceived"):
    """Tang et al.'s sequential utils_eval.generate_null (it loads its own copy of the LM); used only by --check_seq"""
    return utils_eval.generate_null(pred_times, gpt_checkpoint, n)

def last_nuclei(gpt, contexts, nuc_mass, nuc_ratio, chunk=512, check=False):
    """For each context: next-word probs at the last position (as LanguageModel.ps) and the nucleus of
    get_nucleus(probs, nuc_mass, nuc_ratio), computed on GPU (stable sort -> identical tie order)."""
    out = []
    for i in range(0, len(contexts), chunk):
        ids = gpt.get_context_array(contexts[i:i + chunk])
        with torch.no_grad():
            logits = gpt.model(input_ids=ids.to(gpt.device), attention_mask=torch.ones(ids.shape).int().to(gpt.device)).logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
            sp, si = torch.sort(probs, dim=1, descending=True, stable=True)
            keep = sp >= sp[:, :1] * nuc_ratio
            cs = torch.cumsum(sp * keep, dim=1)
            reach = (cs >= nuc_mass) & keep
            first = torch.where(reach.any(1), reach.float().argmax(1), keep.sum(1) - 1)
            nk = torch.minimum(keep.sum(1), first + 1)
            K = int(nk.max())
            sp_c, si_c, nk_c = sp[:, :K].cpu().numpy(), si[:, :K].cpu().numpy(), nk.cpu().numpy()
            full = probs.cpu().numpy() if check else None
        for r in range(len(nk_c)):
            nuc = si_c[r, :nk_c[r]]
            if check:
                ref = get_nucleus(full[r], nuc_mass=nuc_mass, nuc_ratio=nuc_ratio)
                assert list(ref) == list(nuc), ("nucleus mismatch", list(ref)[:10], list(nuc)[:10])
            out.append((nuc, dict(zip(si_c[r, :nk_c[r]].tolist(), sp_c[r, :nk_c[r]].tolist()))))
    return out

def generate_null_batched(pred_times, gpt, lm, n, seed=0, check_steps=20):
    rng = np.random.default_rng(seed)
    decs = [Decoder(pred_times, 2 * config.EXTENSIONS) for _ in range(n)]
    for sample_index in range(len(pred_times)):
        ncontext = decs[0].time_window(sample_index, config.LM_TIME, floor=5)
        if len(decs[0].beam) == 1 and len(decs[0].beam[0].words) == 0:
            nucs_all = [lm.beam_propose(d.beam, ncontext) for d in decs]
        else:
            contexts = [hyp.words[-ncontext:] for d in decs for hyp in d.beam]
            nl = last_nuclei(gpt, contexts, lm.nuc_mass, lm.nuc_ratio, check=sample_index < check_steps)
            nucs_all, k = [], 0
            for d in decs:
                bn = []
                for hyp in d.beam:
                    nuc_ids, pmap = nl[k]; k += 1
                    context = hyp.words[-ncontext:]
                    nuc_words = [lm.model.vocab[i] for i in nuc_ids if i in lm.ids]
                    nuc_words = context_filter(nuc_words, context)
                    bn.append((nuc_words, np.log([pmap[lm.model.word2id[w]] for w in nuc_words])))
                nucs_all.append(bn)
        for d, beam_nucs in zip(decs, nucs_all):
            for c, (hyp, nextensions) in enumerate(d.get_hypotheses()):
                nuc, logprobs = beam_nucs[c]
                if len(nuc) < 1: continue
                likelihoods = rng.random(len(nuc))
                local_extensions = [Hypothesis(parent=hyp, extension=x) for x in zip(nuc, logprobs, [np.zeros(1) for _ in nuc])]
                d.add_extensions(local_extensions, likelihoods, nextensions)
            d.extend(verbose=False)
        if sample_index % 300 == 0: print(f"  null step {sample_index}/{len(pred_times)}", flush=True)
    return [d.beam[0].words for d in decs]

class WERm:
    def score(self, ref, pred):
        return np.array([1.0 if len(r) == 0 else wer(" ".join(r), " ".join(p) if len(p) else "") for r, p in zip(ref, pred)])
class BLEUm:
    def __init__(self, n=1): self.metric = evaluate.load("bleu"); self.n = n
    def score(self, ref, pred):
        out = []
        for r, p in zip(ref, pred):
            ps, rs = " ".join(p), " ".join(r)
            if len(p) == 0 or len(r) == 0: out.append(0.0); continue
            out.append(self.metric._compute(predictions=[ps], references=[[rs]], max_order=self.n)["bleu"])   # same module, no Arrow overhead
        return np.array(out)
class METEORm:
    def __init__(self): self.metric = evaluate.load("meteor")
    def score(self, ref, pred):
        return np.array([self.metric._compute(predictions=[" ".join(p)], references=[" ".join(r)])["meteor"] for r, p in zip(ref, pred)])
class BERTm:
    def __init__(self, idf_sents):
        self.metric = BERTScorer(lang="en", rescale_with_baseline=False, idf=True, idf_sents=idf_sents)
    def score(self, ref, pred):
        return self.metric.score(cands=[" ".join(x) for x in pred], refs=[" ".join(x) for x in ref], batch_size=128)[1].numpy()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True); ap.add_argument("--experiment", default="perceived_speech")
    ap.add_argument("--task", default="wheretheressmoke"); ap.add_argument("--tag", default="")
    ap.add_argument("--null", type=int, default=200); ap.add_argument("--check_seq", type=int, default=0)
    ap.add_argument("--metrics", nargs="+", default=["WER", "BLEU", "METEOR", "BERT"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    T = {}
    t0 = time.time()
    eval_segments = json.load(open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json")))
    pred = np.load(os.path.join(config.RESULT_DIR, a.subject, a.experiment, a.task + a.tag + ".npz"))
    pred_words, pred_times = pred["words"], pred["times"]
    gpt_checkpoint = "imagined" if a.experiment == "imagined_speech" else "perceived"
    gpt, lm = load_lm(gpt_checkpoint)
    out_dir = os.path.join(config.SCORE_DIR, a.subject, a.experiment); os.makedirs(out_dir, exist_ok=True)
    null_path = os.path.join(out_dir, f"{a.task}{a.tag}_nulls{a.null}.json")
    if os.path.exists(null_path):
        null_word_list = json.load(open(null_path))
    else:
        t = time.time(); null_word_list = generate_null_batched(pred_times, gpt, lm, a.null, seed=a.seed); T["null_batched_s"] = time.time() - t
        json.dump(null_word_list, open(null_path, "w"))
    seq_nulls = []
    if a.check_seq:
        t = time.time(); seq_nulls = generate_null_sequential(pred_times, gpt, lm, a.check_seq); T["null_seq_s"] = time.time() - t
    ref = load_transcript(a.experiment, a.task)
    cut = windows(*eval_segments[a.task], config.WINDOW)
    ref_w = segment_data(ref["words"], ref["times"], cut)
    pred_w = segment_data(pred_words, pred_times, cut)
    null_w = [segment_data(nw, pred_times, cut) for nw in null_word_list]
    seq_w = [segment_data(nw, pred_times, cut) for nw in seq_nulls]
    metrics = {}
    if "WER" in a.metrics: metrics["WER"] = WERm()
    if "BLEU" in a.metrics: metrics["BLEU"] = BLEUm(1)
    if "METEOR" in a.metrics: metrics["METEOR"] = METEORm()
    if "BERT" in a.metrics: metrics["BERT"] = BERTm(np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")))
    res, arrs = {"subject": a.subject, "task": a.task, "tag": a.tag, "n_null": len(null_w), "n_windows": len(cut)}, {}
    for m, metric in metrics.items():
        t = time.time()
        ws = metric.score(ref_w, pred_w)
        # BERTScore: score all null windows in one call (fast); others loop
        if m == "BERT":   # chunks of 5 nulls: bert_score caches every unique sentence's token embeddings in CPU RAM
            nws = []
            for c0 in range(0, len(null_w), 5):
                part = null_w[c0:c0 + 5]
                nws.append(metric.score([r for _ in part for r in ref_w], [p for nw in part for p in nw]).reshape(len(part), len(cut)))
            nws = np.concatenate(nws)
        else:
            nws = np.array([metric.score(ref_w, nw) for nw in null_w])
        story_null = nws.mean(1)
        sign = -1.0 if m == "WER" else 1.0   # WER is an error rate: lower is better
        res[m] = {"story": float(ws.mean()), "null_mean": float(story_null.mean()), "null_sd": float(story_null.std()),
                  "z": float(sign * (ws.mean() - story_null.mean()) / story_null.std()),
                  "frac_windows_sig_p05": float(np.mean(sign * (ws - nws.mean(0)) / nws.std(0) > 1.645))}
        if seq_w:
            sq = np.array([metric.score(ref_w, nw) for nw in seq_w]).mean(1)
            res[m]["seq_null_story_mean"] = float(sq.mean()); res[m]["seq_null_story_sd"] = float(sq.std()) if len(sq) > 1 else None
        arrs[m + "_window"] = ws; arrs[m + "_null_window"] = nws.astype(np.float32)
        T[m + "_s"] = time.time() - t
        print(m, json.dumps(res[m]), flush=True)
    res["timing_s"] = T; res["total_s"] = time.time() - t0
    np.savez(os.path.join(out_dir, f"{a.task}{a.tag}_eval.npz"), cutoffs=np.array(cut), **arrs)
    json.dump(res, open(os.path.join(out_dir, f"{a.task}{a.tag}_eval.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
