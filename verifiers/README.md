# Verifiers

Both verifiers score a candidate text against the recording; neither generates text.

- **Encode-then-compare** (`etc_feats.py`, `etc_feats_merge.py`, `etc_pool.py`, `etc_pool600.py`): the decoder's own encoding
  and noise models (fitted per cell by `decoder/tang_fit_e.py`) predict the response to each candidate; the score is the
  likelihood of the recorded window under that prediction. Candidates: the true 20-s segment and a pool of 600
  timing-matched segments of other stories.
- **Contrastive** (`verify.py`, `verify_e.py --preset`): the window's principal components (of the anatomical voxels) are
  projected linearly onto a sentence-embedding space (GTR-T5) and trained with a contrastive loss.
- **Text side** (`prep_text.py`, `prep_text_words.py`): 20-s segments of every story, their embeddings and foils.

Evidence (`analysis/`): log2[(P + 1) / rank] of the true segment among P candidates, minus the same with time-shifted brain
data, so chance is 0 bits; the contrastive verifier is scored on held-out training stories (leave-one-story-out, story-cluster
bootstrap), encode-then-compare and the decoder on the 56 windows of the held-out test story (moving-block bootstrap, blocks
of 3 windows, paired). `analysis/grid.py` computes the learning curves, the training needed for the goal (interpolated in
log10 hours), the noise survived, and the slopes (least squares of bits on log10 hours over n4 ... n82) with bootstrap CIs.
