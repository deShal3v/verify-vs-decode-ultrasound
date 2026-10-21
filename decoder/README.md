# Decoder wrappers and settings

Decoder: Tang, LeBel, Jain & Huth 2023, https://github.com/HuthLab/semantic-decoding at commit
`6b25dd5720ced1c4199cd9b9149061683b77ba6b` (clone it to `$TANG_REPO`; `pipeline/setup_env.sh` does). None of its code is in
this repository: the scripts here import its modules (`config`, `GPT`, `Decoder`, `LanguageModel`, `EncodingModel`,
`StimulusModel`, `utils_stim`, `utils_eval`, `utils_ridge`), and `upstream.py` runs the beam-search loop of
`decoding/run_decoder.py` from the clone after checking the file against that commit (git blob
`d55740a467977c417bed0cd3a45428c12f33a867`). The loop that ran for the paper is AST-identical to that block (checked).

## Settings

- **Decoder parameters**: Tang et al.'s `decoding/config.py`, unchanged: beam width `WIDTH = 200` (the authors' setting, where
  performance levels off), `LM_TIME = 8`, `LM_MASS = 0.9`, `LM_RATIO = 0.1`, `EXTENSIONS = 5`, `NM_ALPHA = 2/3`, GPT layer 9 with
  5 context words, stimulus delays 1-4 TR, 20-s evaluation windows. GPT checkpoint `perceived`.
- **Voxels**: the anatomical set `$WORK_DIR/inputs/<S>/anat_cols.npy` (`voxels/make_anat_cols.py`), intersected with the
  condition's reachable set; all of them are used (`tang_fit_e.py --preset`, no ranking by encoding performance). The noise
  model is fitted on the same voxels.
- **Encoding model**: `tang_fit.py` refits Tang et al.'s model for each condition and amount of training (the first k of the 82
  training stories in their order): GPT features (`get_stim`), bootstrap ridge with 50 bootstraps, 10 alphas
  (logspace(1, 3, 10)), chunks of 40 TRs, per-voxel alphas; the noise model is the leave-one-story-out residual covariance.
  The ridge is solved on the GPU in torch (Gram form, identical to the repository's SVD form); the chunk draws follow the
  repository's chunking with a seeded generator.
- **Frozen word times**: every decode and every null text uses the word times predicted from clean data with all 15.25 h
  (cell `native_r15`: `tang_fit.py --cond native --rung 15` fits the word-rate model on the auditory ROI, `tang_decode.py`
  predicts the times with `predict_word_times(..., starttime=-10)` and decodes; its `decoded.npz` becomes
  `$WORK_DIR/inputs/<S>/native_r15_decoded.npz`). `tang_fit_e.py --frozen` writes these times into each cell's
  `pred_times.npz` (`frozen=True`) and `tang_decode_frozen.py` decodes with them. This is the most favourable timing available
  to the decoder (it does not depend on the cell's noise).
- **Test trial**: repeat 0 of the test story `wheretheressmoke` (the ds004510 single trial Tang et al. decode), with the cell's
  noise; the noise is drawn independently for every story and repeat (`noise/noise_us.py`).
- **Null texts**: the language model's own text on the frozen word times with random likelihoods (Tang et al.'s
  `utils_eval.generate_null` algorithm; `tang_eval.py generate_null_batched` batches the GPT forward passes): 200 with seed 0
  (`tang_eval2.py nulls --n 200`), plus 38 batches of 100 (seeds 1001-1011, 2001-2011, 3001-3008, 4001-4008, the same for every
  participant) merged by `make_nulls_big.py` into 4,000.
- **Evidence in bits** (`tang_bits.py`, `tang_bits_e.py`): on the 56 test windows of 20 s with at least 3 words, BERTScore
  recall (idf-weighted, no baseline rescaling, as in Tang et al.'s evaluate_predictions.py) of the decoded words against the heard words; bits = log2[(N + 1) / (1 + #nulls at least
  as close)] with N = 4,000, minus the same quantity for the nulls themselves (leave-one-null-out), so a brain-free decoder
  scores 0. The goal is log2 K_eff of the clean 15.25-h decoder's median window (`decoder_goal.py`: 4.59 / 4.73 / 5.72 bits).
- One change against the repository, in every wrapper: the Snowball stemmer of the language model's context filter is memoised
  (`functools.lru_cache`), which saves CPU time and does not change outputs.
