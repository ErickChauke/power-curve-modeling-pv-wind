"""Standalone Wind2 power curve model: a parametric power curve formula, GP, and Gradient
Boosting.

No PV-style physics formula: the grid is fully dense, so there is no unsampled-region
extrapolation problem to motivate one. Instead, "curve" is a classic wind-turbine power curve
equation (see wind_power_curve below), a real closed-form formula rather than a fit that only
makes sense next to its training grid, this is the default model since it is the simplest
genuine answer. GP and GBM remain available for the tighter statistical fit (see
notebook.ipynb's Wind2 Model section for the full comparison). Refits all three from the source
grid on import (the dataset is tiny, refitting is instant). Wind direction is circular, so GP/GBM
encode it as (sin, cos) internally rather than raw degrees; the curve formula ignores direction
entirely, since velocity dominates power output far more (see the SHAP analysis in
notebook.ipynb). The documented near-zero negative artifact is clipped to 0 before fitting.
Exposes predict(velocity, direction, model="curve") plus per-model
predict_curve / predict_gp / predict_gbm functions.
"""

import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "FPCs.xlsm"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Capacity from the source workbook's Wind2 sheet title cell (row 0, col 0).
CAPACITY_MW = 86.6


def _parse_grid_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return x_axis, y_axis, power_grid


def _build_features(velocity, direction_deg):
    # Wind direction is circular: 15 deg and 360 deg are adjacent bearings, not 345 deg apart.
    # Encoding it as (sin, cos) instead of raw degrees puts adjacent bearings next to each other
    # in feature space too, so a distance-based model like GP does not see a false discontinuity
    # at the wrap-around point.
    direction_rad = np.deg2rad(direction_deg)
    return np.column_stack([velocity, np.sin(direction_rad), np.cos(direction_rad)])


def wind_power_curve(v, v_cutin, v_rated, p_rated, k):
    """Piecewise cubic-style wind turbine power curve [2]: 0 below cut-in, p_rated above rated
    speed, and a v^k ramp in between."""
    v = np.asarray(v, dtype=float)
    power = np.zeros_like(v)
    ramp = (v >= v_cutin) & (v < v_rated)
    power[ramp] = p_rated * (v[ramp]**k - v_cutin**k) / (v_rated**k - v_cutin**k)
    power[v >= v_rated] = p_rated
    return power


def _fit_curve(velocity, power, capacity_mw):
    # Initial guess and search bounds: cut-in near 3 m/s, rated near 12 m/s, rated power near
    # nameplate capacity, shape exponent between linear (1) and the idealized cubic (up to 10).
    params, _ = curve_fit(
        wind_power_curve, velocity, power,
        p0=[3.0, 12.0, capacity_mw, 3.0],
        bounds=([0.0, 5.0, capacity_mw * 0.8, 1.0], [8.0, 20.0, capacity_mw * 1.2, 10.0]),
    )
    return params


def _fit_models():
    x_axis, y_axis, power_grid = _parse_grid_sheet(DATA_PATH, "Wind2")
    # Clip the documented near-zero negative artifact (measurement noise around zero wind
    # speed, not real negative generation) -- same treatment as the notebook's EDA section.
    power_grid = np.clip(power_grid, 0.0, None)

    vel_mesh, dir_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    df = pd.DataFrame({
        "wind_velocity": vel_mesh.ravel(),
        "wind_direction": dir_mesh.ravel(),
        "power": power_grid.ravel(),
    }).dropna(subset=["power"])

    X = _build_features(df["wind_velocity"].to_numpy(), df["wind_direction"].to_numpy())
    y = df["power"].to_numpy()

    curve_params = _fit_curve(df["wind_velocity"].to_numpy(), y, CAPACITY_MW)

    # Gaussian Process kernel: ConstantKernel scales overall variance, RBF gives each of the
    # three features its own smoothness ("length scale"), WhiteKernel absorbs measurement
    # noise. The ranges are search bounds; sklearn fits the exact values during .fit().
    # n_restarts_optimizer=3 tries 3 starting points to avoid a poor local optimum;
    # random_state=0 makes results reproducible.
    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),  # put all three features on a comparable numeric scale first
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    # Gradient Boosting: builds 300 decision trees one at a time, each correcting the errors
    # left by the trees before it, rather than averaging independent trees like Random Forest.
    # This often fits sharp, non-smooth shapes (a wind power curve's steep cut-in transition)
    # a bit better.
    gbm_model = GradientBoostingRegressor(n_estimators=300, random_state=0).fit(X, y)

    return curve_params, gp_model, gbm_model, x_axis, y_axis, power_grid


CURVE_PARAMS, GP_MODEL, GBM_MODEL, X_AXIS, Y_AXIS, POWER_GRID = _fit_models()


def _predict_generic(model, velocity, direction):
    V, D = np.broadcast_arrays(np.asarray(velocity, dtype=float), np.asarray(direction, dtype=float))
    X = _build_features(V.ravel(), D.ravel())
    pred = model.predict(X).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_curve(velocity, direction=None):
    # direction is accepted (and ignored) so this has the same call signature as predict_gp/gbm.
    V = np.asarray(velocity, dtype=float)
    pred = wind_power_curve(V.ravel(), *CURVE_PARAMS).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_gp(velocity, direction):
    return _predict_generic(GP_MODEL, velocity, direction)


def predict_gbm(velocity, direction):
    return _predict_generic(GBM_MODEL, velocity, direction)


def predict(velocity, direction=None, model="curve"):
    dispatch = {"curve": predict_curve, "gp": predict_gp, "gbm": predict_gbm}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    if model == "curve":
        return predict_curve(velocity)
    return dispatch[model](velocity, direction)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_v, sample_dir = 10.0, 180.0
    print(f"Sample prediction at velocity={sample_v} m/s, direction={sample_dir} deg:")
    for model_name in ["curve", "gp", "gbm"]:
        print(f"  {model_name}: {predict(sample_v, sample_dir, model=model_name):.2f} MW")

    full_vel_mesh, full_dir_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    panels = [
        ("Actual", POWER_GRID),
        ("Curve", predict_curve(full_vel_mesh)),
        ("GP", predict_gp(full_vel_mesh, full_dir_mesh)),
        ("GBM", predict_gbm(full_vel_mesh, full_dir_mesh)),
    ]
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (title, grid_) in zip(axes, panels):
        im = ax.imshow(grid_, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=CAPACITY_MW)
        ax.set_title(f"Wind2 {title}")
        ax.set_xlabel("Wind Direction (deg)")
    axes[0].set_ylabel("Wind Velocity (m/s)")
    fig.colorbar(im, ax=axes, label="Power (MW)", fraction=0.02)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "wind2_response_surface.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved response-surface figure to {out_path}")
