"""Standalone Wind1 power curve model: GP and Gradient Boosting.

No physics formula: the grid is fully dense, so there is no unsampled-region extrapolation
problem to motivate one (see notebook.ipynb's Wind1 Model section). Refits both models from the
source grid on import (the dataset is tiny, refitting is instant). Wind direction is circular, so
it is encoded as (sin, cos) internally rather than fed in as raw degrees. Exposes
predict(velocity, direction, model="gp") plus per-model predict_gp / predict_gbm functions.
"""

import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "FPCs.xlsm"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Capacity from the source workbook's Wind1 sheet title cell (row 0, col 0).
CAPACITY_MW = 102.0


def _parse_grid_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return x_axis, y_axis, power_grid


def _build_features(velocity, direction_deg):
    direction_rad = np.deg2rad(direction_deg)
    return np.column_stack([velocity, np.sin(direction_rad), np.cos(direction_rad)])


def _fit_models():
    x_axis, y_axis, power_grid = _parse_grid_sheet(DATA_PATH, "Wind1")

    vel_mesh, dir_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    df = pd.DataFrame({
        "wind_velocity": vel_mesh.ravel(),
        "wind_direction": dir_mesh.ravel(),
        "power": power_grid.ravel(),
    }).dropna(subset=["power"])

    X = _build_features(df["wind_velocity"].to_numpy(), df["wind_direction"].to_numpy())
    y = df["power"].to_numpy()

    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    gbm_model = GradientBoostingRegressor(n_estimators=300, random_state=0).fit(X, y)

    return gp_model, gbm_model, x_axis, y_axis, power_grid


GP_MODEL, GBM_MODEL, X_AXIS, Y_AXIS, POWER_GRID = _fit_models()


def _predict_generic(model, velocity, direction):
    V, D = np.broadcast_arrays(np.asarray(velocity, dtype=float), np.asarray(direction, dtype=float))
    X = _build_features(V.ravel(), D.ravel())
    pred = model.predict(X).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_gp(velocity, direction):
    return _predict_generic(GP_MODEL, velocity, direction)


def predict_gbm(velocity, direction):
    return _predict_generic(GBM_MODEL, velocity, direction)


def predict(velocity, direction, model="gp"):
    dispatch = {"gp": predict_gp, "gbm": predict_gbm}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    return dispatch[model](velocity, direction)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_v, sample_dir = 10.0, 180.0
    print(f"Sample prediction at velocity={sample_v} m/s, direction={sample_dir} deg:")
    for model_name in ["gp", "gbm"]:
        print(f"  {model_name}: {predict(sample_v, sample_dir, model=model_name):.2f} MW")

    full_vel_mesh, full_dir_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    panels = [
        ("Actual", POWER_GRID),
        ("GP", predict_gp(full_vel_mesh, full_dir_mesh)),
        ("GBM", predict_gbm(full_vel_mesh, full_dir_mesh)),
    ]
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, grid_) in zip(axes, panels):
        im = ax.imshow(grid_, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=CAPACITY_MW)
        ax.set_title(f"Wind1 {title}")
        ax.set_xlabel("Wind Direction (deg)")
    axes[0].set_ylabel("Wind Velocity (m/s)")
    fig.colorbar(im, ax=axes, label="Power (MW)", fraction=0.02)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "wind1_response_surface.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved response-surface figure to {out_path}")
