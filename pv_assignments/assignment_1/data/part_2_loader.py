"""Dataloader functions for part 2 of the assignment."""

import importlib.resources as resources
from typing import Literal

import pandas as pd


def load_data() -> pd.DataFrame:
    """Load datafile for part 2.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    with (
        resources.files("pv_assignments.assignment_1.data")
        .joinpath("34552_risoe_1h_irradiance_2024_v2.csv")
        .open("rb") as f
    ):
        df = pd.read_csv(f)

    return df
