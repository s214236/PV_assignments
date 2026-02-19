"""Task 1 of assignment 1."""

import matplotlib.pyplot as plt
import pandas as pd

from pv_assignments.assignment_1.data.part_1_loader import load_data


def plot(
    dfs: list[pd.DataFrame],
    labels: list[str],
    value_names: list[str],
    peak_wavelengths: list[float] | None = None,
) -> None:
    plt.figure()
    for df, label in zip(dfs, labels):
        for value_name in value_names:
            plt.plot(
                df["Wavelength (nm)"],
                df[value_name],
                label=label,  # TODO: fix labels for more columns
            )

    if peak_wavelengths is not None:
        for x in peak_wavelengths:
            plt.axvline(x, linestyle="--", color="gray", alpha=0.5)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Global to perpendicular plane (W/m2/nm)")
    plt.title("Spectral Irradiance for Different Air Masses")
    plt.legend()
    plt.grid()
    plt.show()


def broadband_irradiance(dfs: list[pd.DataFrame], labels: list[str]) -> None:
    for df, label in zip(dfs, labels):
        irr_broadband = df["Global to perpendicular plane  (W/m2/nm)"].sum() * 10
        print(f"Broadband Irradiance for {label}: {irr_broadband:.2f} W/m²")


def peak_wavelength(dfs: list[pd.DataFrame], labels: list[str]) -> list[float]:
    peak_irr = {}
    for df, label in zip(dfs, labels):
        max_irr = df["Global to perpendicular plane  (W/m2/nm)"].max()
        peak_wavelength = df.loc[
            df["Global to perpendicular plane  (W/m2/nm)"].idxmax(), "Wavelength (nm)"
        ]
        peak_irr[label] = (peak_wavelength, max_irr)
    print("Peak Irradiance and Corresponding Wavelengths:")
    for label, (wavelength, irradiance) in peak_irr.items():
        print(
            f"{label}: Wavelength = {wavelength} nm, Irradiance = {irradiance} W/m²/nm"
        )

    return [peak[0] for peak in peak_irr.values()]


def main() -> None:
    """Main function for part 1 of the assignment."""
    print("\nPart 1-1")
    df_1_5 = load_data("1.5")
    df_3 = load_data("3")
    df_4_5 = load_data("4.5")
    df_6 = load_data("6")
    dfs = [df_1_5, df_3, df_4_5, df_6]
    labels = ["AM 1.5", "AM 3", "AM 4.5", "AM 6"]
    value_names = ["Global to perpendicular plane (W/m2/nm)"]
    broadband_irradiance(dfs, labels)
    peak_wavelengths = peak_wavelength(dfs, labels)
    plot(dfs, labels, value_names, peak_wavelengths=peak_wavelengths)

    print("\nPart 1-2")
    df_1_5 = load_data("1.5")
    dfs = [df_1_5]
    labels = ["Global", "Direct", "Diffuse"]


if __name__ == "__main__":
    main()
