"""Task 2 of assignment 1."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_assignments.assignment_1.data.part_2_loader import load_data
from pv_assignments.utils.panel_irradiation import Panel


def main() -> None:
    """Main function for part 2 of the assignment."""
    print("\nPart 2-1")
    # NOTE: South convention is used. East=-90, West=90, North=+-180, South=0
    # Situation modelled: Morning sun, Panel facing towards evenings sun.
    rho_g = 0.2
    azimuth_panel = 63
    tilt_panel = 41
    panel = Panel(azimuth=azimuth_panel, tilt=tilt_panel, rho_g=rho_g)

    azimuth_sun = -76
    zenith_sun = 63

    angle_of_incidence = panel.calculate_angle_of_incidence(azimuth_sun, zenith_sun)
    print(f"Angle of incidence: {angle_of_incidence}")

    dni = 600
    dhi = 120
    print(f"  GHI: {dhi + dni * np.cos(np.radians(zenith_sun))} W/m^2")  # type: ignore
    direct_irradiance, diffuse_irradiance, ground_reflected_irradiance = (
        panel.calculate_gpoa_components(dhi, dni, azimuth_sun, zenith_sun)
    )
    print(f"  Direct irradiance: {round(direct_irradiance, 2)} W/m^2")  # type: ignore
    print(f"  Diffuse irradiance: {round(diffuse_irradiance, 2)} W/m^2")  # type: ignore
    print(
        f"  Ground reflected irradiance: {round(ground_reflected_irradiance, 2)} W/m^2"  # type: ignore
    )

    print("\nPart 2-2")

    df = load_data()
    df.rename(
        columns={"SolarElevation": "Zenith", "SolarAzimuth": "Azimuth"}, inplace=True
    )
    df["Zenith"] = 90 - df["Zenith"]  # Convert elevation to zenith
    df["Azimuth"] = df["Azimuth"] - 180  # Convert to south convention

    panel_horizontal = Panel(azimuth=0, tilt=0, rho_g=rho_g)
    panel_south = Panel(azimuth=0, tilt=90, rho_g=rho_g)
    panel_south_tilted = Panel(azimuth=0, tilt=45, rho_g=rho_g)
    panel_west_tilted = Panel(azimuth=90, tilt=45, rho_g=rho_g)

    df_monthly_insolation = panel_horizontal.calculate_monthly_insolation(df).rename(
        columns={"Insolation": "Insolation_horizontal"}
    )
    df_monthly_insolation = df_monthly_insolation.merge(
        panel_south.calculate_monthly_insolation(df).rename(
            columns={"Insolation": "Insolation_south"}
        ),
        on="DateTime",
    )
    df_monthly_insolation = df_monthly_insolation.merge(
        panel_south_tilted.calculate_monthly_insolation(df).rename(
            columns={"Insolation": "Insolation_south_tilted"}
        ),
        on="DateTime",
    )
    df_monthly_insolation = df_monthly_insolation.merge(
        panel_west_tilted.calculate_monthly_insolation(df).rename(
            columns={"Insolation": "Insolation_west_tilted"}
        ),
        on="DateTime",
    )
    plt.figure()
    plt.plot(
        df_monthly_insolation["DateTime"],
        df_monthly_insolation["Insolation_horizontal"],
        label="Horizontal",
    )
    plt.plot(
        df_monthly_insolation["DateTime"],
        df_monthly_insolation["Insolation_south"],
        label="South",
    )
    plt.plot(
        df_monthly_insolation["DateTime"],
        df_monthly_insolation["Insolation_south_tilted"],
        label="South Tilted",
    )
    plt.plot(
        df_monthly_insolation["DateTime"],
        df_monthly_insolation["Insolation_west_tilted"],
        label="West Tilted",
    )
    plt.xlabel("Date")
    plt.ylabel("Monthly Insolation (Wh/m²)")
    plt.title("Comparison of Monthly Insolation for Different Panel Orientations")
    plt.legend()
    plt.show()

    print("  Annual insolation values:")
    print(f"    Horizontal: {df_monthly_insolation['Insolation_horizontal'].sum()}")
    print(f"    South: {df_monthly_insolation['Insolation_south'].sum()}")
    print(f"    South Tilted: {df_monthly_insolation['Insolation_south_tilted'].sum()}")
    print(f"    West Tilted: {df_monthly_insolation['Insolation_west_tilted'].sum()}")

    print("  Diffuse fraction values:")
    print(
        f"    Horizontal: {panel_horizontal.calculate_diffuse_fraction(df['DHI'], df['DNI'], df['Azimuth'], df['Zenith']).mean()}"  # type: ignore
    )
    print(
        f"    South: {panel_south.calculate_diffuse_fraction(df['DHI'], df['DNI'], df['Azimuth'], df['Zenith']).mean()}"  # type: ignore
    )
    print(
        f"    South Tilted: {panel_south_tilted.calculate_diffuse_fraction(df['DHI'], df['DNI'], df['Azimuth'], df['Zenith']).mean()}"  # type: ignore
    )
    print(
        f"    West Tilted: {panel_west_tilted.calculate_diffuse_fraction(df['DHI'], df['DNI'], df['Azimuth'], df['Zenith']).mean()}"  # type: ignore
    )

    print("\nPart 2-3")
    print("\nPart 2-4")
    print("  Yearly transposition factor:")
    print(
        f"    South: {df_monthly_insolation['Insolation_south'].sum() / df_monthly_insolation['Insolation_horizontal'].sum()}"  # type: ignore
    )
    print(
        f"    South Tilted: {df_monthly_insolation['Insolation_south_tilted'].sum() / df_monthly_insolation['Insolation_horizontal'].sum()}"  # type: ignore
    )
    print(
        f"    West Tilted: {df_monthly_insolation['Insolation_west_tilted'].sum() / df_monthly_insolation['Insolation_horizontal'].sum()}"  # type: ignore
    )


if __name__ == "__main__":
    main()
