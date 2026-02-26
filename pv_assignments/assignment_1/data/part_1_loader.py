"""Dataloader functions for part 1 of the assignment."""

import importlib.resources as resources
from typing import Literal

import pandas as pd


def load_data(
    am: str | Literal["1.5", "3", "4.5", "6"],
    water_vapor: str | None | Literal["0", "0.71", "2.13"] = None,
) -> pd.DataFrame:
    """Load datafiles for task one, based on the AM and water_vapor values.

    Args:
        am (str | "1.5" | "3" | "4.5" | "6"): Number of atmospheres the solar arrays have to passe through.
            Can only be the string representation of 1.5, 3, 4.5, or 6.
        water_vapor (str | None | "0" | "0.71" | "2.13", optional): Water vapor content in the atmosphere.
            Can only be the string representation of 0, 0.71, or 2.13.
            If None, the DataFrame is from 1.42 cm of water vapor. Defaults to None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    # Create the file name based on the provided parameters
    file_name = "Spectrum"
    file_name += f"_AM{'_'.join(am.split('.'))}"
    if water_vapor is not None:
        file_name += f"_WP{'_'.join(water_vapor.split('.'))}"

    with (
        resources.files("pv_assignments.assignment_1.data")
        .joinpath(f"{file_name}.xlsx")
        .open("rb") as f
    ):
        df = pd.read_excel(f, sheet_name="Spectral irradiance")

    return df
