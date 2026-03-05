"""Task 2 of assignment 1."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pv_assignments.assignment_1.data.part_2_loader import load_data


class Panel:
    """Class representing a solar panel."""

    def __init__(self, azimuth: float, tilt: float, rho_g: float = 0.2) -> None:
        """Initialize the panel with its azimuth, tilt and ground reflectance.

        Args:
            azimuth (float): Azimuth angle of the panel (degrees).
            tilt (float): Tilt angle of the panel (degrees).
            rho_g (float, optional): Ground reflectance (default is 0.2).
        """
        self.azimuth = azimuth
        self.tilt = tilt
        self.rho_g = rho_g

    def calculate_angle_of_incidence(
        self, azimuth_sun: float | np.ndarray, zenith_sun: float | np.ndarray
    ) -> float | np.ndarray:
        """Calculate the angle of incidence of the sun on the panel.

        Args:
            azimuth_sun (float | np.ndarray): Azimuth angle of the sun (degrees).
            zenith_sun (float | np.ndarray): Zenith angle of the sun (degrees).

        Returns:
            float | np.ndarray: Angle of incidence (degrees).
        """
        angle_of_incidence = np.degrees(
            np.arccos(
                np.cos(np.radians(zenith_sun)) * np.cos(np.radians(self.tilt))
                + np.sin(np.radians(zenith_sun))
                * np.sin(np.radians(self.tilt))
                * np.cos(np.radians(azimuth_sun - self.azimuth))
            )
        )
        return angle_of_incidence

    def calculate_gpoa_components(
        self,
        dhi: float | np.ndarray,
        dni: float | np.ndarray,
        angle_of_incidence: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Calculate the components of the global plane of array irradiance.

        Args:
            dhi (float | np.ndarray): Diffuse horizontal irradiance
            dni (float | np.ndarray): Direct normal irradiance
            angle_of_incidence (float | np.ndarray): Angle of incidence of the sun on the panel

        Returns:
            tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]: Direct irradiance, diffuse irradiance, ground reflected irradiance.
        """
        direct_irradiance = np.maximum(dni * np.cos(np.radians(angle_of_incidence)), 0)
        diffuse_irradiance = dhi * (1 + np.cos(np.radians(self.tilt))) / 2
        ground_reflected_irradiance = (
            (dhi + direct_irradiance)
            * self.rho_g
            * (1 - np.cos(np.radians(self.tilt)))
            / 2
        )
        return direct_irradiance, diffuse_irradiance, ground_reflected_irradiance

    def calculate_gpoa(
        self,
        dhi: float | np.ndarray,
        dni: float | np.ndarray,
        azimuth_sun: float | np.ndarray,
        zenith_sun: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate the global plane of array irradiance.

        Args:
            dhi (float | np.ndarray): Diffuse horizontal irradiance
            dni (float | np.ndarray): Direct normal irradiance
            azimuth_sun (float | np.ndarray): Azimuth angle of the sun
            zenith_sun (float | np.ndarray): Zenith angle of the sun
        Returns:
            float | np.ndarray: Global plane of array irradiance.
        """
        angle_of_incidence = self.calculate_angle_of_incidence(azimuth_sun, zenith_sun)
        direct_irradiance, diffuse_irradiance, ground_reflected_irradiance = (
            self.calculate_gpoa_components(dhi, dni, angle_of_incidence)
        )
        gpoa = direct_irradiance + diffuse_irradiance + ground_reflected_irradiance
        return gpoa

    def calculate_daily_insolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the daily insolation on the panel.

        Args:
            df (pd.DataFrame): DataFrame containing the irradiance data.
                Required columns: "DateTime", "DHI", "DNI", "Azimuth", "Zenith".
                where Zenith and Azimuth are the sun's position at the given DateTime.

        Returns:
            pd.DataFrame: DataFrame with the daily insolation values.
        """
        df["GPOA"] = self.calculate_gpoa(
            dhi=df["DHI"],
            dni=df["DNI"],
            azimuth_sun=df["Azimuth"],
            zenith_sun=df["Zenith"],
        )
        df_daily = df.resample("D", on="DateTime").mean().reset_index()
        df_daily["Insolation"] = df_daily["GPOA"] * 24  # Assuming 24 hours per day
        return df_daily[["DateTime", "Insolation"]]

    def calculate_monthly_insolation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate the monthly insolation on the panel.

        Args:
            df (pd.DataFrame): DataFrame containing the irradiance data.
                Required columns: "DateTime", "DHI", "DNI", "Azimuth", "Zenith".
                where Zenith and Azimuth are the sun's position at the given DateTime.

        Returns:
            pd.DataFrame: DataFrame with the monthly insolation values.
        """
        df_daily = self.calculate_daily_insolation(df)
        df_monthly = df_daily.resample("MS", on="DateTime").mean().reset_index()
        return df_monthly[["DateTime", "Insolation"]]

    def calculate_diffuse_fraction(
        self,
        dhi: float | np.ndarray,
        dni: float | np.ndarray,
        azimuth_sun: float | np.ndarray,
        zenith_sun: float | np.ndarray,
    ) -> float | np.ndarray:
        """Calculate the diffuse contribution of the irradiance.

        Args:
            dhi (float | np.ndarray): Diffuse horizontal irradiance
            dni (float | np.ndarray): Direct normal irradiance
            azimuth_sun (float | np.ndarray): Azimuth angle of the sun
            zenith_sun (float | np.ndarray): Zenith angle of the sun
        Returns:
            float | np.ndarray: Diffuse contribution.
        """
        direct, diffuse, ground_reflected = self.calculate_gpoa_components(
            dhi, dni, self.calculate_angle_of_incidence(azimuth_sun, zenith_sun)
        )
        return diffuse / (direct + diffuse + ground_reflected)


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
    direct_irradiance, diffuse_irradiance, ground_reflected_irradiance = (
        panel.calculate_gpoa_components(dhi, dni, angle_of_incidence)
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
