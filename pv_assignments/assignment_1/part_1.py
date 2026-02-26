"""Task 1 of assignment 1."""

import matplotlib.pyplot as plt
import pandas as pd

from pv_assignments.assignment_1.data.part_1_loader import load_data


def plot(
    dfs: list[pd.DataFrame],
    labels: list[str],
    value_names: list[str],
    peak_wavelengths: bool = False,
) -> None:
    """Plot spectral irradiance for multiple dataframes.

    Args:
        dfs (list[pd.DataFrame]): DataFrames containing the spectral irradiance data.
        labels (list[str]): Labels for each DataFrame.
        value_names (list[str]): Names of the columns to plot.
        peak_wavelengths (bool, optional): Whether to mark peak wavelengths with vertical lines.
            Defaults to False.
    """
    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels, strict=True):
        for value_name in value_names:
            new_label = f"{label} - {value_name}"
            plt.plot(
                df["Wavelength (nm)"],
                df[value_name],
                label=new_label,
            )
            if peak_wavelengths:
                peak = peak_wavelength([df], [label], [value_name], silent=True)[
                    new_label
                ]
                plt.axvline(
                    x=peak["wavelength"], color="black", linestyle="--", alpha=0.5
                )

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Global to perpendicular plane (W/m2/nm)")
    plt.title("Spectral Irradiance for Different Air Masses")
    plt.legend()
    plt.grid()
    plt.show()


def broadband_irradiance(
    dfs: list[pd.DataFrame], labels: list[str], value_names: list[str]
) -> dict[str, float]:
    """Calculate the total broadband irradiance (integration).

    Args:
        dfs (list[pd.DataFrame]): List of dataframes
        labels (list[str]): Names of each dataframe
        value_names (list[str]): Names of the columns to integrate
    """
    broadband_irradiance_values = {}
    for df, label in zip(dfs, labels, strict=False):
        for value_name in value_names:
            new_label = f"{label} - {value_name}"
            irr_broadband = df[value_name].sum() * 10
            broadband_irradiance_values[new_label] = irr_broadband
            print(f"Broadband Irradiance for {new_label}: {irr_broadband:.2f} W/m²")

    return broadband_irradiance_values


def peak_wavelength(
    dfs: list[pd.DataFrame],
    labels: list[str],
    value_names: list[str],
    silent: bool = True,
) -> dict[str, dict[str, float]]:
    """Calculate peak wavelengths.

    Args:
        dfs (list[pd.DataFrame]): List of dataframes.
        labels (list[str]): Names of dataframes.
        value_names (list[str]): Names of the columns to investigate.
        silent (bool, optional): Whether to print the results. Defaults to True.

    Returns:
        dict[str, dict[str, float]]: Nested dict with label -> ("wavelength" | "irradiance").
    """
    peak_irradiance_values = {}
    for df, label in zip(dfs, labels, strict=False):
        for value_name in value_names:
            new_label = f"{label} - {value_name}"
            max_irr = df[value_name].max()
            peak_wavelength = df.loc[df[value_name].idxmax(), "Wavelength (nm)"]
            peak_irradiance_values[new_label] = {
                "wavelength": peak_wavelength,
                "irradiance": max_irr,
            }
    if not silent:
        print("Peak Irradiance and Corresponding Wavelengths:")
        for label, data in peak_irradiance_values.items():
            print(
                f"{label}: Wavelength = {data['wavelength']} nm, Irradiance = {data['irradiance']} W/m²/nm"
            )

    return peak_irradiance_values


def main() -> None:
    """Main function for part 1 of the assignment."""
    print("\nPart 1-1")
    dfs = [
        load_data("1.5"),
        load_data("3"),
        load_data("4.5"),
        load_data("6"),
    ]
    labels = ["AM 1.5", "AM 3", "AM 4.5", "AM 6"]
    value_names = ["Global to perpendicular plane  (W/m2/nm)"]
    broadband_irradiance(dfs, labels, value_names)
    peak_wavelength(dfs, labels, value_names)
    plot(dfs, labels, value_names, peak_wavelengths=True)

    print("\nPart 1-2")
    dfs = [load_data("1.5")]
    labels = ["AM 1.5"]
    value_names = [
        "Direct to horizontal plane (W/m2/nm)",
        "Diffuse to horizontal plane (W/m2/nm)",
        "Global to horizontal plane  (W/m2/nm)",
    ]
    plot(dfs=dfs, labels=labels, value_names=value_names)
    broadband = broadband_irradiance(dfs, labels, value_names)
    diffuse_fraction = (
        broadband["AM 1.5 - Diffuse to horizontal plane (W/m2/nm)"]
        / broadband["AM 1.5 - Global to horizontal plane  (W/m2/nm)"]
    )
    print(f"Diffuse fraction for AM 1.5: {diffuse_fraction:.2f}")

    print("\nPart 1-3")
    dfs = [
        load_data("1.5", water_vapor="0"),
        load_data("1.5", water_vapor="0.71"),
        load_data("1.5"),
        load_data("1.5", water_vapor="2.13"),
    ]
    labels = [
        "AM 1.5 - Water Vapor 0.00",
        "AM 1.5 - Water Vapor 0.71",
        "AM 1.5 - Water Vapor 1.42",
        "AM 1.5 - Water Vapor 2.13",
    ]
    value_names = ["Global to perpendicular plane  (W/m2/nm)"]
    plot(dfs=dfs, labels=labels, value_names=value_names)


if __name__ == "__main__":
    main()
