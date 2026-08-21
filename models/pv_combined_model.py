"""Standalone PV-combined power curve model: physics formula, GP, and Random Forest.

Fits on PV1 and PV2 pooled together, with plant as a one-hot feature for GP/RF (the physics
formula has no plant term, see notebook.ipynb's PV-combined Model section for why). Refits all
three models from the source grid on import (the dataset is tiny, refitting is instant). Exposes
predict(irradiance, module_temp, plant, model="physics") plus per-model
predict_physics / predict_gp / predict_rf functions.
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "FPCs.xlsm"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Capacity from the source workbook's PV1/PV2 sheet title cells (row 0, col 0); both are 75 MW.
CAPACITY_MW = 75.0


def _parse_grid_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return x_axis, y_axis, power_grid


def _grid_to_df(path, sheet_name, plant):
    x_axis, y_axis, power_grid = _parse_grid_sheet(path, sheet_name)
    irr_mesh, temp_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    df = pd.DataFrame({
        "irradiance": irr_mesh.ravel(),
        "module_temp": temp_mesh.ravel(),
        "power": power_grid.ravel(),
    }).dropna(subset=["power"])
    df["plant"] = plant
    return df, x_axis, y_axis, power_grid


def _fit_models():
    pv1_df, x_axis, y_axis, pv1_grid = _grid_to_df(DATA_PATH, "PV1", "PV1")
    pv2_df, _, _, pv2_grid = _grid_to_df(DATA_PATH, "PV2", "PV2")
    df = pd.concat([pv1_df, pv2_df], ignore_index=True)

    # is_pv2: a one-hot "is this row from PV2?" feature (0 for PV1 rows, 1 for PV2 rows), how
    # the pooled GP/RF models are told which plant each row came from. The physics formula
    # below deliberately does NOT get this feature, one shared beta0/beta1 fit on the pooled
    # data is the cleanest test of whether a single physics formula generalizes across plants.
    is_pv2 = (df["plant"] == "PV2").astype(float)
    X = np.column_stack([df["irradiance"], df["module_temp"], is_pv2])
    X_physics = np.column_stack([df["irradiance"], df["irradiance"] * (df["module_temp"] - 25)])
    y = df["power"].to_numpy()

    # fit_intercept=False: zero irradiance should mean zero power, so no constant term.
    physics_model = LinearRegression(fit_intercept=False).fit(X_physics, y)

    # Gaussian Process kernel: ConstantKernel scales overall variance, RBF gives each feature
    # (including is_pv2) its own smoothness ("length scale"), WhiteKernel absorbs measurement
    # noise. The ranges are search bounds; sklearn fits the exact values during .fit().
    # n_restarts_optimizer=3 tries 3 starting points to avoid a poor local optimum;
    # random_state=0 makes results reproducible.
    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),  # put all three features on a comparable numeric scale first
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    # Random Forest: averages 300 decision trees, each trained on a random resample of the
    # data, for a smoother and more stable prediction than any single tree.
    rf_model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)

    return physics_model, gp_model, rf_model, x_axis, y_axis, pv1_grid, pv2_grid


PHYSICS_MODEL, GP_MODEL, RF_MODEL, X_AXIS, Y_AXIS, PV1_GRID, PV2_GRID = _fit_models()


def _predict_generic(model, build_features, irradiance, module_temp, plant):
    G, Tm = np.broadcast_arrays(np.asarray(irradiance, dtype=float), np.asarray(module_temp, dtype=float))
    is_pv2 = np.where(np.broadcast_to(np.asarray(plant), G.shape) == "PV2", 1.0, 0.0)
    X = build_features(G.ravel(), Tm.ravel(), is_pv2.ravel())
    pred = model.predict(X).reshape(G.shape)
    return float(pred) if pred.shape == () else pred


def predict_physics(irradiance, module_temp, plant="PV1"):
    return _predict_generic(
        PHYSICS_MODEL, lambda G, Tm, is_pv2: np.column_stack([G, G * (Tm - 25)]),
        irradiance, module_temp, plant,
    )


def predict_gp(irradiance, module_temp, plant="PV1"):
    return _predict_generic(
        GP_MODEL, lambda G, Tm, is_pv2: np.column_stack([G, Tm, is_pv2]),
        irradiance, module_temp, plant,
    )


def predict_rf(irradiance, module_temp, plant="PV1"):
    return _predict_generic(
        RF_MODEL, lambda G, Tm, is_pv2: np.column_stack([G, Tm, is_pv2]),
        irradiance, module_temp, plant,
    )


def predict(irradiance, module_temp, plant="PV1", model="physics"):
    if plant not in ("PV1", "PV2"):
        raise ValueError(f"plant must be 'PV1' or 'PV2', got {plant!r}")
    dispatch = {"physics": predict_physics, "gp": predict_gp, "rf": predict_rf}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    return dispatch[model](irradiance, module_temp, plant)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_G, sample_Tm = 800.0, 40.0
    print(f"Sample prediction at G={sample_G} W/m^2, Tm={sample_Tm} C:")
    for plant in ["PV1", "PV2"]:
        for model_name in ["physics", "gp", "rf"]:
            pred = predict(sample_G, sample_Tm, plant=plant, model=model_name)
            print(f"  plant={plant}, model={model_name}: {pred:.2f} MW")

    full_irr_mesh, full_temp_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]
    grids = {"PV1": PV1_GRID, "PV2": PV2_GRID}

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for row, plant in enumerate(["PV1", "PV2"]):
        panels = [
            ("Actual", grids[plant]),
            ("Physics (shared)", predict_physics(full_irr_mesh, full_temp_mesh, plant=plant)),
            ("GP", predict_gp(full_irr_mesh, full_temp_mesh, plant=plant)),
            ("RF", predict_rf(full_irr_mesh, full_temp_mesh, plant=plant)),
        ]
        for ax, (title, grid_) in zip(axes[row], panels):
            im = ax.imshow(grid_, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=CAPACITY_MW)
            ax.set_title(f"{plant} {title}")
            ax.set_xlabel("Module Temperature (C)")
        axes[row][0].set_ylabel("Irradiance (W/m^2)")
    fig.colorbar(im, ax=axes, label="Power (MW)", fraction=0.02)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "pv_combined_response_surface.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved response-surface figure to {out_path}")
