# PV & Wind Power Curve Modeling

Erick Chauke

Physics-informed and data-driven power curve models for two PV plants and two wind plants, fit
from source power-curve grids (input files, gitignored).

There are two notebooks here:

- **`notebook_simple.ipynb`** (start here): one plain equation per plant, a physics formula for
  the two PV plants and a classic wind-turbine power curve formula for the two wind plants. Each
  is a genuine closed-form equation, computed directly from a handful of fitted numbers, not a
  fit to the grid at prediction time. Includes a raw-data heatmap, a 3D surface plot, and an
  accuracy plot per plant, in plain language throughout.
- **`notebook.ipynb`** (the detailed backup): the full six-model comparison behind that choice,
  physics/Gaussian Process/Random Forest for PV, Gaussian Process/Gradient Boosting for wind,
  plus pooled combined models, response-surface and residual heatmaps, SHAP explanations, and
  leave-one-plant-out transferability checks. This is the evidence that the simple formulas in
  `notebook_simple.ipynb` are the right choice, not a claim taken on faith.

This is a proof of concept built on the aggregated power-curve grids available today (a few
hundred to just over a thousand cells per plant), not raw plant telemetry. See
`notebook.ipynb`'s Limitations & Future Work section for what richer data (raw time-series
telemetry, denser sampling, more plants per technology) would unlock beyond this proof of
concept.

## Overview

For each of PV1, PV2, Wind1, Wind2, `notebook_simple.ipynb` fits one formula per plant: the
standard PV temperature-coefficient model for the two PV plants, and the standard cubic wind
turbine power curve model for the two wind plants (neither plant has run its own turbine
spec-sheet test, so this is fit to the grid the same way the PV formula is). `notebook.ipynb`
additionally compares each of those against Gaussian Process and tree-ensemble models, and adds
two pooled "combined" models per technology (plant as an added feature) with a
leave-one-plant-out transferability check.

## Project structure

```
power-curve-modeling-pv-wind/
  notebook_simple.ipynb   One formula per plant, plain language, start here
  notebook_simple.html    Static export of the simple notebook (dark-mode aware)
  notebook.ipynb          Full six-model analysis, the technical backup
  notebook.html           Static export of the full notebook (dark-mode aware)
  README.md               This file
  .gitignore
  models/
    pv1_model.py           predict(irradiance, module_temp, model="physics"|"gp"|"rf")
    pv2_model.py            same interface as pv1_model.py
    pv_combined_model.py    predict(irradiance, module_temp, plant="PV1"|"PV2", model=...)
    wind1_model.py          predict(velocity, direction, model="curve"|"gp"|"gbm")
    wind2_model.py          same interface as wind1_model.py
    wind_combined_model.py  predict(velocity, direction, plant="Wind1"|"Wind2", model=...)
  outputs/                 Figures from notebook.ipynb, source-name-prefixed
  outputs_simple/          Figures from notebook_simple.ipynb, source-name-prefixed
  data/                    Gitignored: input workbook and planning notes
```

## Pointing this at a new input file

Every model refits from the source grid on import, there are no pickled models to go stale. To
point the project at a new workbook:

1. Place the workbook in `data/` (gitignored). It must have the same layout documented in
   `notebook.ipynb`'s Data Loading & Grid Parsing section: four sheets (PV1, PV2, Wind1, Wind2),
   each a 2D grid with the capacity label in the title cell (row 0, col 0), the x-axis values
   across row 1 from column 2 onward, the y-axis values down column 1 from row 2 onward, and the
   power grid filling the rest.
2. Update the `DATA_PATH` constant in both notebooks' Configuration cells, and in each
   `models/*.py` script, to point at the new file.
3. Update the `CAPACITY_MW` values (both notebooks' config cells and each script) if the new
   plants have different nameplate capacities, they are currently hardcoded from each sheet's
   title cell as documented in the source comment next to them.
4. Re-run whichever notebook top to bottom.

## Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy`, `openpyxl`, `scikit-learn`, `shap` (only needed for
`notebook.ipynb`), `jupyter`/`nbconvert`.

## How to run

- **Notebooks**: open either `.ipynb` file in VS Code or JupyterLab and run all cells top to
  bottom (needs the extensions `ms-toolsai.jupyter` and `ms-toolsai.jupyter-renderers` for the
  LaTeX formulas to render, and `cweijan.vscode-office` for this file to render correctly in
  VS Code).
- **Static HTML**: open either `.html` file directly in any browser, no environment needed, both
  follow your OS/browser dark-mode setting.
- **Standalone scripts**: each `models/*.py` file is self-contained and refits its model(s) on
  import (the datasets are tiny, refitting is instant). Import `predict` from the relevant module,
  or run the script directly (`python models/pv1_model.py`) to see a sample prediction and
  regenerate that model's response-surface figure into `outputs/`.
