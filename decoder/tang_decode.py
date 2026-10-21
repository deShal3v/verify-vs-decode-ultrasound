"""Phase B: Tang 2023 decoder, exactly as decoding/run_decoder.py (via Phase A tang_run.py), with a refitted cell model.
Differences from tang_run.py: the encoding model is the compact Phase B file (weights for the selected voxels only;
mathematically identical to weights[:, voxels]); the response is the cell's test_resp.npy (test repeat 0 plus the
cell's noise); word times come from the cell's refitted word-rate model (same predict_word_rate/predict_word_times).
The beam-search loop itself is executed from the pinned clone of Tang et al.'s repository (upstream.py), not copied.
Usage: python tang_decode.py --cell_dir $DATA_DIR/models/S1/native_r15 [--resp other.npy] [--out decoded.npz]"""
import os as _os, sys as _sys; _d = _os.path.dirname(_os.path.abspath(__file__)); _sys.path.insert(0, next(p for p in (_d, _os.path.dirname(_d), _os.path.dirname(_os.path.dirname(_d))) if _os.path.exists(_os.path.join(p, 'vvd_paths.py')))); import vvd_paths as VP  # DATA_DIR / WORK_DIR
import os, sys, time, json, argparse
from functools import lru_cache
REPO = VP.TANG_CODE; sys.path.insert(0, REPO)
import numpy as np, torch
if os.environ.get("TANG_MEM_FRAC"): torch.cuda.set_per_process_memory_fraction(float(os.environ["TANG_MEM_FRAC"]))
import config
import LanguageModel as LMmod
LMmod.stemmer.stem = lru_cache(maxsize=None)(LMmod.stemmer.stem)
from GPT import GPT
from Decoder import Decoder, Hypothesis
from LanguageModel import LanguageModel
from EncodingModel import EncodingModel
from StimulusModel import StimulusModel, get_lanczos_mat, affected_trs, LMFeatures
from utils_stim import predict_word_rate, predict_word_times
import upstream

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--cell_dir", required=True); p.add_argument("--resp", default=None); p.add_argument("--out", default=None)
    args = p.parse_args(); t0 = time.time(); cd = args.cell_dir
    out = args.out or os.path.join(cd, "decoded.npz")
    resp = np.load(args.resp or os.path.join(cd, "test_resp.npy")).astype(np.float64)
    with open(os.path.join(config.DATA_LM_DIR, "perceived", "vocab.json")) as f: gpt_vocab = json.load(f)
    with open(os.path.join(config.DATA_LM_DIR, "decoder_vocab.json")) as f: decoder_vocab = json.load(f)
    gpt = GPT(path=os.path.join(config.DATA_LM_DIR, "perceived", "model"), vocab=gpt_vocab, device=config.GPT_DEVICE)
    features = LMFeatures(model=gpt, layer=config.GPT_LAYER, context_words=config.GPT_WORDS)
    lm = LanguageModel(gpt, decoder_vocab, nuc_mass=config.LM_MASS, nuc_ratio=config.LM_RATIO)
    wrm = np.load(os.path.join(cd, "word_rate_model_auditory.npz"), allow_pickle=True)
    em = np.load(os.path.join(cd, "encoding_model.npz"))
    vox = em["voxels"]; assert "wheretheressmoke" not in em["stories"]
    tr_stats = em["tr_stats"]; word_stats = em["word_stats"]
    enc = EncodingModel(np.ascontiguousarray(resp[:, vox]), em["weights_sel"].astype(np.float64), np.arange(len(vox)), em["noise_model"], device=config.EM_DEVICE)
    enc.set_shrinkage(config.NM_ALPHA)
    word_rate = predict_word_rate(resp, wrm["weights"], wrm["voxels"], wrm["mean_rate"])
    word_times, tr_times = predict_word_times(word_rate, resp, starttime=-10)
    pt = np.load(os.path.join(cd, "pred_times.npz")); assert np.allclose(pt["word_times"], word_times) or args.resp
    lanczos_mat = get_lanczos_mat(word_times, tr_times)
    t_load = time.time() - t0
    print(f"[{cd}] loaded in {t_load:.0f}s; {len(word_times)} predicted words, resp {resp.shape}", flush=True)
    t1 = time.time()
    # beam search: the decode loop of Tang et al.'s decoding/run_decoder.py, read from the pinned clone and run unchanged (upstream.py)
    decoder = upstream.run_decode_loop(dict(word_times=word_times, lanczos_mat=lanczos_mat, tr_stats=tr_stats, word_stats=word_stats, lm=lm,
                                            features=features, em=enc, config=config, Decoder=Decoder, Hypothesis=Hypothesis,
                                            StimulusModel=StimulusModel, affected_trs=affected_trs))
    decoder.save(out[:-4] if out.endswith(".npz") else out)
    t_dec = time.time() - t1
    json.dump({"cell_dir": cd, "n_words": int(len(word_times)), "load_s": t_load, "decode_s": t_dec, "total_s": time.time() - t0},
              open(out.replace(".npz", "_timing.json"), "w"))
    print(f"DONE decode {t_dec/60:.1f} min", flush=True)
