# Results

## Which run the paper uses

The paper's numbers and Figs. 2-3 come from the merged grid of the GPU runs Phase E + E2 + E3 + E4
(`verifiers/analysis/grid.py`, output `GRID_E4_s7.json`; slopes over the 7 common training amounts n4 ... n82, `SLOPE_SET=7`):

- Phase E: clean data and uniform noise x4, x16, x63 (`native`, `m6c`, `m12c`, `m18c`), amounts n1 ... n82 (verifiers), n4 ... n82 (decoder);
- Phase E2: the same conditions at n56 and n73;
- Phase E4: the ultrasound conditions with the link budget per 2-s scan, K = 2 (`usL90k2` one probe optimistic, `usB90k2` two
  probes optimistic, `usL50k2` one probe median, `usL50k2u6/u12/u18` median with added gain), and the decoder at x63 (`m18c`);
- Phase E3 (coded excitation) is merged into the grid file but none of its conditions is used in the ISBI paper; the
  Phase E ultrasound cells at K = 10 (`usL90`, `usB90`, `usL50`) are superseded by the K = 2 cells.

Mapping to the figure keys (`make_fig_data.py`): usL90 -> usL90k2, usB90 -> usB90k2, usL50 -> usL50k2, gains -> usL50k2u6/u12/u18.

## Learning curves

Full learning curves (PNG, every participant, readout and condition) are added at the final sync from `figures/data/fig2_data_e4.json`.

## Within-story control

Placeholder: the table is added before release (`verifiers/within_story_control.py`).
