"""Phase B: LM-only nulls and scoring for one Tang cell (Phase A tang_eval.py functions, unchanged metrics).
  nulls : python tang_eval2.py nulls --cell_dir D --n 100 [--seed s]     -> D/nulls{n}.json   (needs only pred_times.npz)
  score : python tang_eval2.py score --cell_dir D --n 100                -> D/eval.json, D/eval_windows.npz
Nulls use the cell's predicted word times (identical to the decoder's saved times). BERTScore of the nulls is
computed in chunks of 5 nulls (Phase A OOM fix)."""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from tang_eval import (config, load_lm, generate_null_batched, load_transcript, windows, segment_data, WERm, BLEUm, METEORm, BERTm)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["nulls", "score"]); ap.add_argument("--cell_dir", required=True)
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred", default=None); ap.add_argument("--tag", default="")
    a = ap.parse_args(); cd = a.cell_dir; t0 = time.time()
    null_path = os.path.join(cd, f"nulls{a.n}.json")
    if a.mode == "nulls":
        times = np.load(os.path.join(cd, "pred_times.npz"))["word_times"]
        gpt, lm = load_lm("perceived")
        nl = generate_null_batched(times, gpt, lm, a.n, seed=a.seed)
        json.dump(nl, open(null_path, "w")); json.dump({"null_s": time.time() - t0, "n": a.n}, open(null_path.replace(".json", "_timing.json"), "w"))
        print("nulls done", time.time() - t0, flush=True); sys.exit(0)
    task = "wheretheressmoke"; exp = "perceived_speech"
    eval_segments = json.load(open(os.path.join(config.DATA_TEST_DIR, "eval_segments.json")))
    pred = np.load(a.pred or os.path.join(cd, "decoded.npz")); pred_words, pred_times = pred["words"], pred["times"]
    null_word_list = json.load(open(null_path))
    ref = load_transcript(exp, task); cut = windows(*eval_segments[task], config.WINDOW)
    ref_w = segment_data(ref["words"], ref["times"], cut); pred_w = segment_data(pred_words, pred_times, cut)
    null_w = [segment_data(nw, pred_times, cut) for nw in null_word_list]
    metrics = {"WER": WERm(), "BLEU": BLEUm(1), "METEOR": METEORm(), "BERT": BERTm(np.load(os.path.join(config.DATA_TEST_DIR, "idf_segments.npy")))}
    res, arrs, T = {"cell_dir": cd, "n_null": len(null_w), "n_windows": len(cut)}, {}, {}
    for m, metric in metrics.items():
        t = time.time(); ws = metric.score(ref_w, pred_w)
        if m == "BERT":
            nws = []
            for c0 in range(0, len(null_w), 5):
                part = null_w[c0:c0 + 5]
                nws.append(metric.score([r for _ in part for r in ref_w], [p for nw in part for p in nw]).reshape(len(part), len(cut)))
            nws = np.concatenate(nws)
        else:
            nws = np.array([metric.score(ref_w, nw) for nw in null_w])
        story_null = nws.mean(1); sign = -1.0 if m == "WER" else 1.0
        res[m] = {"story": float(ws.mean()), "null_mean": float(story_null.mean()), "null_sd": float(story_null.std()),
                  "z": float(sign * (ws.mean() - story_null.mean()) / story_null.std()),
                  "frac_windows_sig_p05": float(np.mean(sign * (ws - nws.mean(0)) / nws.std(0) > 1.645)),
                  "story_rank_p": float((1 + np.sum(sign * story_null >= sign * ws.mean())) / (1 + len(story_null)))}
        arrs[m + "_window"] = ws; arrs[m + "_null_window"] = nws.astype(np.float32); T[m + "_s"] = time.time() - t
        print(m, json.dumps(res[m]), flush=True)
    res["timing_s"] = T; res["total_s"] = time.time() - t0
    np.savez(os.path.join(cd, f"eval{a.tag}_windows.npz"), cutoffs=np.array(cut), **arrs)
    json.dump(res, open(os.path.join(cd, f"eval{a.tag}.json"), "w"), indent=1)
