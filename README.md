# PV & Wind Power Curve Modeling

Erick Chauke

Physics-informed and data-driven power curve models for two PV plants and two wind plants, fit
from source power-curve grids (input files, gitignored). Six standalone models: PV1, PV2,
PV-combined, Wind1, Wind2, Wind-combined.

This is a proof of concept built on the aggregated power-curve grids available today (a few
hundred to just over a thousand cells per plant), not raw plant telemetry. It demonstrates a
complete modeling pipeline end to end (grid parsing, physics and ML fits, cross-validation,
response-surface and residual diagnostics, SHAP explanations, deployable `predict()` functions)
and surfaces real findings on the data at hand. See the notebook's Limitations & Future Work
section for what richer data (raw time-series telemetry, denser sampling, more plants per
technology) would unlock beyond this proof of concept.

## Overview

For each of PV1, PV2, and Wind1, Wind2, a standalone model is fit against that plant's own data,
plus a combined model per technology (PV-combined, Wind-combined) that pools both plants with
plant as an added feature. PV models compare a physics-informed temperature-coefficient formula
against Gaussian Process and Random Forest regressors, since the PV grids have real unsampled
regions the physics formula must extrapolate into more safely than the ML models. Wind models are
data-driven only (Gaussian Process and Gradient Boosting), since the wind grids are fully dense
and there is no extrapolation problem to motivate a physics formula. Every model gets
response-surface and residual heatmaps and a SHAP feature-importance summary, and the two
combined models additionally get a leave-one-plant-out transferability check.

## Project structure

```
power-curve-modeling-pv-wind/
  notebook.ipynb          Full analysis, built and committed one subsection at a time
  notebook.html           Static export of the notebook (dark-mode aware), open in any browser
  README.md               This file
  .gitignore
  models/
    pv1_model.py           predict(irradiance, module_temp, model="physics"|"gp"|"rf")
    pv2_model.py            same interface as pv1_model.py
    pv_combined_model.py    predict(irradiance, module_temp, plant="PV1"|"PV2", model=...)
    wind1_model.py          predict(velocity, direction, model="gp"|"gbm")
    wind2_model.py          same interface as wind1_model.py
    wind_combined_model.py  predict(velocity, direction, plant="Wind1"|"Wind2", model=...)
  outputs/                 Committed figures, source-name-prefixed (e.g. pv1_response_surface.png)
  data/                    Gitignored: input workbook and planning notes
```

## Pointing this at a new input file

Every model refits from the source grid on import, there are no pickled models to go stale. To
point the project at a new workbook:

1. Place the workbook in `data/` (gitignored). It must have the same layout documented in the
   notebook's Data Loading & Grid Parsing section: four sheets (PV1, PV2, Wind1, Wind2), each a
   2D grid with the capacity label in the title cell (row 0, col 0), the x-axis values across
   row 1 from column 2 onward, the y-axis values down column 1 from row 2 onward, and the power
   grid filling the rest.
2. Update the `DATA_PATH` constant in the notebook's Configuration cell, and in each
   `models/*.py` script, to point at the new file.
3. Update the `CAPACITY_MW` values (notebook config cell and each script) if the new plants have
   different nameplate capacities, they are currently hardcoded from each sheet's title cell as
   documented in the source comment next to them.
4. Re-run the notebook top to bottom.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `openpyxl`, `scikit-learn`, `shap`, `jupyter`/`nbconvert`.

## How to run

- **Notebook**: open `notebook.ipynb` in VS Code or JupyterLab and run all cells top to bottom
  (needs the extensions `ms-toolsai.jupyter` and `ms-toolsai.jupyter-renderers` for the LaTeX
  formulas to render, and `cweijan.vscode-office` for this file to render correctly in VS Code).
- **Static HTML**: open `notebook.html` directly in any browser, no environment needed, it
  follows your OS/browser dark-mode setting.
- **Standalone scripts**: each `models/*.py` file is self-contained and refits its model(s) on
  import (the datasets are tiny, refitting is instant). Import `predict` from the relevant module,
  or run the script directly (`python models/pv1_model.py`) to see a sample prediction and
  regenerate that model's response-surface figure into `outputs/`.
