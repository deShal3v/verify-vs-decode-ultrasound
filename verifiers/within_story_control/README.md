# Within-story control

Does the contrastive verifier only tell stories apart? Here the wrong options come from the SAME story: every 20-s window of
the held-out story at 2-s steps whose start is at least 20 s from the true window's start (about 323 options per window),
against the paper's pool of every 20-s segment of the other stories (5,370). Contrastive verifier, clean and -12 dB uniform
noise, 15.25 h of training, one leave-one-story-out model per training story with the paper's seeds; the other-story score
reproduces the paper's value to 3 decimals in all 6 cells. Results: `results/within_story_control.csv`, `results/results.md`.

| script | does |
|---|---|
| `ws_text.py` | the same-story candidate windows (GTR embeddings of the 10 sub-windows, built as the text side builds a segment) |
| `verify_ws.py` | the verifier exactly as `verify_e.py`, scored with both pools |
| `run_ws.sh` | one participant on one GPU node (about 35 min on an NVIDIA L4 including setup) |
| `ws_analysis.py` | rank bits per window, story-cluster bootstrap with the grid's draws -> `within_story_control.csv` |

The GPU scripts run from the staged code directory (`pipeline/stage_code.sh` copies them); `ws_analysis.py` runs on the CPU:
`python verifiers/within_story_control/ws_analysis.py --rescue $WORK_DIR/rescue/ws --grid results/GRID_E4_s7.json --out results`.
