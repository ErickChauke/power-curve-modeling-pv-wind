"""Standalone Wind-combined power curve model: GP and Gradient Boosting.

Fits on Wind1 and Wind2 pooled together, with plant as a one-hot feature for both models (see
notebook.ipynb's Wind-combined Model section). Refits both models from the source grid on import
(the dataset is tiny, refitting is instant). Wind direction is circular, so it is encoded as
(sin, cos) internally. The documented Wind2 negative-value artifact is clipped to 0 before
fitting. Exposes predict(velocity, direction, plant, model="gp") plus per-model
predict_gp / predict_gbm functions.
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
    direction_rad = np.deg2rad(direction_deg)
    return np.column_stack([velocity, np.sin(direction_rad), np.cos(direction_rad), is_wind2])


def _fit_models():
    wind1_df, x_axis, y_axis, wind1_grid = _grid_to_df(DATA_PATH, "Wind1", "Wind1")
    wind2_df, _, _, wind2_grid = _grid_to_df(DATA_PATH, "Wind2", "Wind2", clip=True)
    df = pd.concat([wind1_df, wind2_df], ignore_index=True)

    is_wind2 = (df["plant"] == "Wind2").astype(float)
    X = _build_features(df["wind_velocity"].to_numpy(), df["wind_direction"].to_numpy(), is_wind2.to_numpy())
    y = df["power"].to_numpy()

    gp_kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF([1.0, 1.0, 1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(1.0, (1e-5, 1e2))
    gp_model = make_pipeline(
        StandardScaler(),
        GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0),
    ).fit(X, y)

    gbm_model = GradientBoostingRegressor(n_estimators=300, random_state=0).fit(X, y)

    return gp_model, gbm_model, x_axis, y_axis, wind1_grid, wind2_grid


GP_MODEL, GBM_MODEL, X_AXIS, Y_AXIS, WIND1_GRID, WIND2_GRID = _fit_models()


def _predict_generic(model, velocity, direction, plant):
    V, D = np.broadcast_arrays(np.asarray(velocity, dtype=float), np.asarray(direction, dtype=float))
    is_wind2 = np.where(np.broadcast_to(np.asarray(plant), V.shape) == "Wind2", 1.0, 0.0)
    X = _build_features(V.ravel(), D.ravel(), is_wind2.ravel())
    pred = model.predict(X).reshape(V.shape)
    return float(pred) if pred.shape == () else pred


def predict_gp(velocity, direction, plant="Wind1"):
    return _predict_generic(GP_MODEL, velocity, direction, plant)


def predict_gbm(velocity, direction, plant="Wind1"):
    return _predict_generic(GBM_MODEL, velocity, direction, plant)


def predict(velocity, direction, plant="Wind1", model="gp"):
    if plant not in ("Wind1", "Wind2"):
        raise ValueError(f"plant must be 'Wind1' or 'Wind2', got {plant!r}")
    dispatch = {"gp": predict_gp, "gbm": predict_gbm}
    if model not in dispatch:
        raise ValueError(f"model must be one of {list(dispatch)}, got {model!r}")
    return dispatch[model](velocity, direction, plant)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    sample_v, sample_dir = 10.0, 180.0
    print(f"Sample prediction at velocity={sample_v} m/s, direction={sample_dir} deg:")
    for plant in ["Wind1", "Wind2"]:
        for model_name in ["gp", "gbm"]:
            pred = predict(sample_v, sample_dir, plant=plant, model=model_name)
            print(f"  plant={plant}, model={model_name}: {pred:.2f} MW")

    full_vel_mesh, full_dir_mesh = np.meshgrid(Y_AXIS, X_AXIS, indexing="ij")
    extent = [X_AXIS.min(), X_AXIS.max(), Y_AXIS.min(), Y_AXIS.max()]
    grids = {"Wind1": WIND1_GRID, "Wind2": WIND2_GRID}
    cap = max(CAPACITY_MW.values())

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, plant in enumerate(["Wind1", "Wind2"]):
        panels = [
            ("Actual", grids[plant]),
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
