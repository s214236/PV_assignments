"""Dataloader functions for part 1 of the assignment."""

import importlib.resources as resources
from typing import Literal

import pandas as pd


def load_data(
    am: str | Literal["1.5", "3", "4.5", "6"],
    water_vapor: str | None | Literal["0", "2", "4"] = None,
) -> pd.DataFrame:
    """Load datafiles for task one, based on the AM and water_vapor values.

    Args:
        am (str | "1.5" | "3" | "4.5" | "6"): Number of atmospheres the solar arrays have to passe through.
            Can only be the string representation of 1.5, 3, 4.5, or 6.
        water_vapor (str | None | "0" | "2" | "4", optional): Water vapor content in the atmosphere.
            Can only be the string representation of 0, 2, or 4. Defaults to None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    file_name = f"Spectrum_AM{int(am)}"
    if am != int(am):
        file_name += f"_{str(am).split('.')[-1]}"
    if water_vapor is not None:
        file_name += f"_WP{water_vapor}"

    with (
        resources.files("pv_assignments.assignment_1.data")
        .joinpath(f"{file_name}.xlsx")
        .open("rb") as f
    ):
        df = pd.read_excel(f, sheet_name="Spectral irradiance")

    return df
