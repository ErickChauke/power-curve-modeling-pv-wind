"""Standalone PV2 power curve model: physics formula, GP, and Random Forest.

Refits all three models from the source grid on import (the dataset is tiny, refitting is
instant). Exposes predict(irradiance, module_temp, model="physics") plus per-model
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

# Capacity from the source workbook's PV2 sheet title cell (row 0, col 0).
CAPACITY_MW = 75.0


def _parse_grid_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return x_axis, y_axis, power_grid


def _fit_models():
    x_axis, y_axis, power_grid = _parse_grid_sheet(DATA_PATH, "PV2")

    irr_mesh, temp_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    df = pd.DataFrame({
        "irradiance": irr_mesh.ravel(),
        "module_temp": temp_mesh.ravel(),
        "power": power_grid.ravel(),
    }).dropna(subset=["power"])

    X = np.column_stack([df["irradiance"], df["module_temp"]])
    X_physics = np.column_stack([df["irradiance"], df["irradiance"] * (df["module_temp"] - 25)])
    y = df["power"].to_numpy()

    # fit_intercept=False: zero irradiance should mean zero power, so no constant term.
    physics_model = LinearRegression(fit_intercept=False).fit(X_physics, y)

    # Gaussian Process kernel: ConstantKernel scales overall variance, RBF gives each feature
    # its own smoothness ("length scale"), WhiteKernel absorbs measurement noise. The ranges are
    # search bounds; sklearn fits the exact values to this dataset. n_restarts_optimizer=3 tries
    # 3 starting points to avoid a poor local optimum; random_state=0 makes results reproducible.
    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),  # put both features on a comparable numeric scale first
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    # Random Forest: averages 300 decision trees, each trained on a random resample of the
    # data, for a smoother and more stable prediction than any single tree.
    rf_model = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)

    return physics_model, gp_model, rf_model, x_axis, y_axis, power_grid


PHYSICS_MODEL, GP_MODEL, RF_MODEL, X_AXIS, Y_AXIS, POWER_GRID = _fit_models()


def _predict_generic(model, build_features, irradiance, module_temp):
    G, Tm = np.broadcast_arrays(np.asarray(irradiance, dtype=float), np.asarray(module_temp, dtype=float))
    X = build_features(G.ravel(), Tm.ravel())
    pred = model.predict(X).reshape(G.shape)
    return float(pred) if pred.shape == () else pred


def predict_physics(irradiance, module_temp):
    return _predict_generic(PHYSICS_MODEL, lambda G, Tm: np.column_stack([G, G * (Tm - 25)]), irradiance, module_temp)


def predict_gp(irradiance, module_temp):
    return _predict_generic(GP_MODEL, lambda G, Tm: np.column_stack([G, Tm]), irradiance, module_temp)


def predict_rf(irradiance, module_temp):
    return _predict_generic(RF_MODEL, lambda G, Tm: np.column_stack([G, Tm]), irradiance, module_temp)


def predict(irradiance, module_temp, model="physics"):
    dispatch = {"physics": predict_physics, "gp": predict_gp, "rf": predict_rf}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    return dispatch[model](irradiance, module_temp)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_G, sample_Tm = 800.0, 40.0
    print(f"Sample prediction at G={sample_G} W/m^2, Tm={sample_Tm} C:")
    for model_name in ["physics", "gp", "rf"]:
        print(f"  {model_name}: {predict(sample_G, sample_Tm, model=model_name):.2f} MW")

    full_irr_mesh, full_temp_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    panels = [
        ("Actual", POWER_GRID),
        ("Physics", predict_physics(full_irr_mesh, full_temp_mesh)),
        ("GP", predict_gp(full_irr_mesh, full_temp_mesh)),
        ("RF", predict_rf(full_irr_mesh, full_temp_mesh)),
    ]
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (title, grid_) in zip(axes, panels):
        im = ax.imshow(grid_, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=CAPACITY_MW)
        ax.set_title(f"PV2 {title}")
        ax.set_xlabel("Module Temperature (C)")
    axes[0].set_ylabel("Irradiance (W/m^2)")
    fig.colorbar(im, ax=axes, label="Power (MW)", fraction=0.02)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "pv2_response_surface.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved response-surface figure to {out_path}")
