"""Standalone Wind-combined power curve model: a parametric power curve formula, GP, and
Gradient Boosting.

GP and GBM fit on Wind1 and Wind2 pooled together, with plant as a one-hot feature (see
notebook.ipynb's Wind-combined Model section). The "curve" model is different: rated power,
cut-in speed, and rated speed are physical properties of one specific turbine fleet, there is no
meaningful "shared" version of them the way a GP kernel or an RF split can be shared with a plant
flag. So "curve" here is simply each plant's own formula (see wind1_model.py / wind2_model.py),
refit here for a consistent interface, this is the default model since it is the simplest
genuine answer for a given plant. Refits all three from the source grid on import (the dataset is
tiny, refitting is instant). Wind direction is circular, so GP/GBM encode it as (sin, cos)
internally; the curve formula ignores direction entirely, since velocity dominates power output
far more (see the SHAP analysis in notebook.ipynb). The documented Wind2 negative-value artifact
is clipped to 0 before fitting. Exposes predict(velocity, direction, plant, model="curve") plus
per-model predict_curve / predict_gp / predict_gbm functions.
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

# Capacities from the source workbook's Wind1/Wind2 sheet title cells (row 0, col 0).
CAPACITY_MW = {"Wind1": 102.0, "Wind2": 86.6}


def _parse_grid_sheet(path, sheet_name):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return x_axis, y_axis, power_grid


def _grid_to_df(path, sheet_name, plant, clip=False):
    x_axis, y_axis, power_grid = _parse_grid_sheet(path, sheet_name)
    if clip:
        # Clip the documented near-zero negative artifact (measurement noise around zero wind
        # speed, not real negative generation) -- same treatment as the notebook's EDA section.
        power_grid = np.clip(power_grid, 0.0, None)
    vel_mesh, dir_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    df = pd.DataFrame({
        "wind_velocity": vel_mesh.ravel(),
        "wind_direction": dir_mesh.ravel(),
        "power": power_grid.ravel(),
    }).dropna(subset=["power"])
    df["plant"] = plant
    return df, x_axis, y_axis, power_grid


def _build_features(velocity, direction_deg, is_wind2):
    # Wind direction is circular: 15 deg and 360 deg are adjacent bearings, not 345 deg apart.
    # Encoding it as (sin, cos) instead of raw degrees puts adjacent bearings next to each other
    # in feature space too, so a distance-based model like GP does not see a false discontinuity
    # at the wrap-around point. is_wind2 is a one-hot "is this row from Wind2?" feature (0 for
    # Wind1 rows, 1 for Wind2 rows), how the pooled models are told which plant each row is from.
    direction_rad = np.deg2rad(direction_deg)
    return np.column_stack([velocity, np.sin(direction_rad), np.cos(direction_rad), is_wind2])


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
    wind1_df, x_axis, y_axis, wind1_grid = _grid_to_df(DATA_PATH, "Wind1", "Wind1")
    wind2_df, _, _, wind2_grid = _grid_to_df(DATA_PATH, "Wind2", "Wind2", clip=True)
    df = pd.concat([wind1_df, wind2_df], ignore_index=True)

    curve_params = {
        "Wind1": _fit_curve(wind1_df["wind_velocity"].to_numpy(), wind1_df["power"].to_numpy(), CAPACITY_MW["Wind1"]),
        "Wind2": _fit_curve(wind2_df["wind_velocity"].to_numpy(), wind2_df["power"].to_numpy(), CAPACITY_MW["Wind2"]),
    }

    is_wind2 = (df["plant"] == "Wind2").astype(float)
    X = _build_features(df["wind_velocity"].to_numpy(), df["wind_direction"].to_numpy(), is_wind2.to_numpy())
    y = df["power"].to_numpy()

    # Gaussian Process kernel: ConstantKernel scales overall variance, RBF gives each of the
    # four features (including is_wind2) its own smoothness ("length scale"), WhiteKernel
    # absorbs measurement noise. The ranges are search bounds; sklearn fits the exact values
    # during .fit(). n_restarts_optimizer=3 tries 3 starting points to avoid a poor local
    # optimum; random_state=0 makes results reproducible.
    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),  # put all four features on a comparable numeric scale first
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    # Gradient Boosting: builds 300 decision trees one at a time, each correcting the errors
    # left by the trees before it, rather than averaging independent trees like Random Forest.
    # This often fits sharp, non-smooth shapes (a wind power curve's steep cut-in transition)
    # a bit better.
    gbm_model = GradientBoostingRegressor(n_estimators=300, random_state=0).fit(X, y)

    return curve_params, gp_model, gbm_model, x_axis, y_axis, wind1_grid, wind2_grid


CURVE_PARAMS, GP_MODEL, GBM_MODEL, X_AXIS, Y_AXIS, WIND1_GRID, WIND2_GRID = _fit_models()


def _predict_generic(model, velocity, direction, plant):
    V, D = np.broadcast_arrays(np.asarray(velocity, dtype=float), np.asarray(direction, dtype=float))
    is_wind2 = np.where(np.broadcast_to(np.asarray(plant), V.shape) == "Wind2", 1.0, 0.0)
    X = _build_features(V.ravel(), D.ravel(), is_wind2.ravel())
    pred = model.predict(X).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_curve(velocity, direction=None, plant="Wind1"):
    # direction is accepted (and ignored) so this has the same call signature as predict_gp/gbm.
    V = np.asarray(velocity, dtype=float)
    pred = wind_power_curve(V.ravel(), *CURVE_PARAMS[plant]).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_gp(velocity, direction, plant="Wind1"):
    return _predict_generic(GP_MODEL, velocity, direction, plant)


def predict_gbm(velocity, direction, plant="Wind1"):
    return _predict_generic(GBM_MODEL, velocity, direction, plant)


def predict(velocity, direction=None, plant="Wind1", model="curve"):
    if plant not in ("Wind1", "Wind2"):
        raise ValueError(f"plant must be 'Wind1' or 'Wind2', got {plant!r}")
    dispatch = {"curve": predict_curve, "gp": predict_gp, "gbm": predict_gbm}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    if model == "curve":
        return predict_curve(velocity, plant=plant)
    return dispatch[model](velocity, direction, plant)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_v, sample_dir = 10.0, 180.0
    print(f"Sample prediction at velocity={sample_v} m/s, direction={sample_dir} deg:")
    for plant in ["Wind1", "Wind2"]:
        for model_name in ["curve", "gp", "gbm"]:
            pred = predict(sample_v, sample_dir, plant=plant, model=model_name)
            print(f"  plant={plant}, model={model_name}: {pred:.2f} MW")

    full_vel_mesh, full_dir_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]
    grids = {"Wind1": WIND1_GRID, "Wind2": WIND2_GRID}
    cap = max(CAPACITY_MW.values())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for row, plant in enumerate(["Wind1", "Wind2"]):
        panels = [
            ("Actual", grids[plant]),
            ("Curve", predict_curve(full_vel_mesh, plant=plant)),
            ("GP", predict_gp(full_vel_mesh, full_dir_mesh, plant=plant)),
            ("GBM", predict_gbm(full_vel_mesh, full_dir_mesh, plant=plant)),
        ]
        for ax, (title, grid_) in zip(axes[row], panels):
            im = ax.imshow(grid_, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=cap)
            ax.set_title(f"{plant} {title}")
            ax.set_xlabel("Wind Direction (deg)")
        axes[row][0].set_ylabel("Wind Velocity (m/s)")
    fig.colorbar(im, ax=axes, label="Power (MW)", fraction=0.02)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "wind_combined_response_surface.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved response-surface figure to {out_path}")
