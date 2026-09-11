"""Shared helpers for reading FPCs.xlsm-style grid sheets and looking up values in them.

No fitting, no statistical modeling: parsing a sheet into arrays, and finding the closest
already-measured grid cell to a requested input. This module is imported (not copy-pasted) by
every notebook in notebooks/, so there is exactly one place that knows how the source sheets are
laid out.
"""

import numpy as np
import pandas as pd


def parse_grid_sheet(path, sheet_name):
    """Positional parse of one FPCs.xlsm-style grid sheet.

    Each sheet is a 2D grid, not row-per-observation data, so this reads by cell position
    rather than trusting pandas' automatic header detection, which the merged/label cells break:
      - row 0, col 0: the capacity title cell, e.g. "Installed Capacity: 75 MW".
      - row 1, columns 2 onward: the x-axis values.
      - column 1, rows 2 onward: the y-axis values.
      - rows 2+, columns 2+: the power grid itself.

    Returns (capacity_label, x_axis, y_axis, power_grid).
    """
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, engine="openpyxl")
    capacity_label = df.iloc[0, 0]
    x_axis = df.iloc[1, 2:].astype(float).to_numpy()
    y_axis = df.iloc[2:, 1].astype(float).to_numpy()
    power_grid = df.iloc[2:, 2:].astype(float).to_numpy()
    return capacity_label, x_axis, y_axis, power_grid


def grid_to_table(x_axis, y_axis, power_grid, x_name, y_name):
    """Reshape a 2D grid into a tidy table: one row per (x, y, power) cell.

    This is the "lookup table" in its plainest form: every measured combination as its own row.
    Rows where power was never measured (a NaN cell in the source grid) are dropped, since there
    is nothing to look up there.
    """
    y_mesh, x_mesh = np.meshgrid(y_axis, x_axis, indexing="ij")
    table = pd.DataFrame({y_name: y_mesh.ravel(), x_name: x_mesh.ravel(), "power": power_grid.ravel()})
    return table.dropna(subset=["power"]).reset_index(drop=True)


def circular_distance(a, b, period=360.0):
    """Shortest distance between two angles on a circle of the given period.

    Plain wind direction (15 deg and 360 deg, say) is 345 deg apart by simple subtraction, but
    those bearings are actually adjacent, 15 deg apart the other way around the compass. This
    picks whichever direction around the circle is shorter.
    """
    diff = np.abs(np.asarray(a) - np.asarray(b)) % period
    return np.minimum(diff, period - diff)


def nearest_lookup(table, x_name, y_name, x_query, y_query, circular_x=False, x_period=360.0):
    """Find the row in `table` whose (x, y) is closest to (x_query, y_query), and return it.

    This is the whole "model": no fitting, just "which measurement do we already have that is
    closest to what was asked for". Distance is measured independently on each axis and
    combined the ordinary (Euclidean) way; if `circular_x` is set, the x-axis distance wraps
    around at `x_period` instead of growing without bound (see circular_distance above), which
    matters for wind direction but not for irradiance or module temperature.
    """
    x_dist = circular_distance(table[x_name], x_query, x_period) if circular_x else (table[x_name] - x_query)
    y_dist = table[y_name] - y_query
    total_dist = np.sqrt(x_dist**2 + y_dist**2)
    return table.loc[total_dist.idxmin()]
