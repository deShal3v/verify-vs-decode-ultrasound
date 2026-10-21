"""Use Tang et al.'s own beam-search loop without copying it into this repository.

The decode loop of decoding/run_decoder.py (HuthLab/semantic-decoding, commit 6b25dd5720ced1c4199cd9b9149061683b77ba6b) is read
from the clone at $TANG_REPO when a decode starts, checked against that commit's file hash, and executed in the caller's namespace.
The block runs from "decoder = Decoder(word_times, config.WIDTH)" to "decoder.extend(verbose = False)": it builds the
StimulusModel and extends every beam hypothesis word by word, scoring the LM's proposals with the encoding model (em.prs).
The decodes of the paper ran the same statements (AST-identical, apart from a progress print every 100 words); only the
word times and the encoding model given to the loop differ from run_decoder.py (see tang_decode.py, tang_decode_frozen.py)."""
import os, hashlib, textwrap
import vvd_paths as VP

RUN_DECODER_BLOB_SHA1 = "d55740a467977c417bed0cd3a45428c12f33a867"     # git blob of decoding/run_decoder.py at the pinned commit
START, END = "decoder = Decoder(word_times, config.WIDTH)", "decoder.extend(verbose = False)"


def _blob_sha1(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def decode_loop_source(path=None):
    path = path or os.path.join(VP.TANG_CODE, "run_decoder.py"); raw = open(path, "rb").read()
    if _blob_sha1(raw) != RUN_DECODER_BLOB_SHA1:
        raise RuntimeError(f"{path} is not run_decoder.py of HuthLab/semantic-decoding@{VP.TANG_COMMIT} (git blob {_blob_sha1(raw)})")
    lines = raw.decode().splitlines()
    i0 = next(i for i, l in enumerate(lines) if l.strip() == START)
    i1 = next(i for i, l in enumerate(lines) if i > i0 and l.strip() == END)
    return textwrap.dedent("\n".join(lines[i0:i1 + 1])) + "\n"


def run_decode_loop(ns):
    """ns: dict with word_times, lanczos_mat, tr_stats, word_stats, lm, features, em (an EncodingModel), config, Decoder,
    Hypothesis, StimulusModel, affected_trs. Runs the loop in ns and returns the finished Decoder (ns['decoder'])."""
    exec(compile(decode_loop_source(), "run_decoder.py:decode-loop", "exec"), ns)
    return ns["decoder"]
