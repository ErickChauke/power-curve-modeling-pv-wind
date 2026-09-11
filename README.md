# PV & Wind Power Curve Data

Erick Chauke

An exploration of four power-curve grids (two PV plants, two wind plants), fit from source
power-curve grids (input files, gitignored). No fitted models, no machine learning: the source
grids already are the answer, each notebook here is about representing and looking values up in
that data clearly, not predicting beyond it.

## Notebooks

- **`notebooks/01_data_exploration.ipynb`**: load and parse all four grids, raw heatmaps, basic
  descriptive statistics (fill rate, min/max/mean power), the PV missingness pattern, and the
  Wind2 negative-value data-quality check.
- **`notebooks/02_pv_power_lookup.ipynb`**: for PV1 and PV2, a heatmap, a 3D view, the lookup
  table itself, and a nearest-measurement lookup function (given an irradiance and module
  temperature, find the closest thing actually measured).
- **`notebooks/03_wind_power_lookup.ipynb`**: the same for Wind1 and Wind2, with wind direction
  handled as a circular axis (15 deg and 360 deg are neighboring bearings, not far apart) so the
  lookup finds the true nearest measurement near the wrap-around point.

## Project structure

```
power-curve-modeling-pv-wind/
  notebooks/
    01_data_exploration.ipynb
    02_pv_power_lookup.ipynb
    03_wind_power_lookup.ipynb
  src/
    grid_lookup.py          Shared sheet parsing + nearest-cell lookup, no fitting
  outputs/                  Figures, source-name-prefixed
  data/                     Gitignored: input workbook and planning notes
  README.md                 This file
  .gitignore
```

## Pointing this at a new input file

There is nothing to retrain, a lookup only ever reads the sheet directly. To point the project
at a new workbook:

1. Place the workbook in `data/` (gitignored). It must have the same layout documented in
   `notebooks/01_data_exploration.ipynb`'s Load All Four Grids section, and in
   `src/grid_lookup.py`'s `parse_grid_sheet` docstring: four sheets (PV1, PV2, Wind1, Wind2),
   each a 2D grid with the capacity label in the title cell (row 0, col 0), the x-axis values
   across row 1 from column 2 onward, the y-axis values down column 1 from row 2 onward, and the
   power grid filling the rest.
2. Update the `DATA_PATH` constant in each notebook's Setup cell to point at the new file.
3. Update the `CAPACITY_MW` values in `02_pv_power_lookup.ipynb` and
   `03_wind_power_lookup.ipynb` if the new plants have different nameplate capacities, they are
   currently hardcoded from each sheet's title cell.
4. Re-run the notebooks top to bottom, in order (01, then 02 and 03, which do not depend on each
   other).

## Dependencies

`numpy`, `pandas`, `matplotlib`, `openpyxl`, `jupyter`/`nbconvert`. Nothing else, there is no
statistical or machine-learning library anywhere in this project.

## How to run

- **Notebooks**: open any file in `notebooks/` in VS Code or JupyterLab and run all cells top to
  bottom (needs `ms-toolsai.jupyter` and `ms-toolsai.jupyter-renderers`, plus `cweijan.vscode-office`
  for this file to render correctly in VS Code).
- **Static HTML**: each notebook also has a `.html` export alongside it in `notebooks/`, open it
  directly in any browser, no environment needed; it follows your OS/browser dark-mode setting.
- **The lookup itself**: `src/grid_lookup.py` is importable on its own (see any notebook's Setup
  cell for the two-line import), if you just want `parse_grid_sheet`, `grid_to_table`, or
  `nearest_lookup` in your own script.
