"""Class for irradiance calculations on a solar panel based on its orientation and the sun's position."""

import numpy as np
import pandas as pd


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
        azimuth_sun: float | np.ndarray,
        zenith_sun: float | np.ndarray,
    ) -> tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]:
        """Calculate the components of the global plane of array irradiance.

        Args:
            dhi (float | np.ndarray): Diffuse horizontal irradiance
            dni (float | np.ndarray): Direct normal irradiance
            azimuth_sun (float | np.ndarray): Azimuth angle of the sun
            zenith_sun (float | np.ndarray): Zenith angle of the sun

        Returns:
            tuple[float | np.ndarray, float | np.ndarray, float | np.ndarray]: Direct irradiance, diffuse irradiance, ground reflected irradiance.
        """
        angle_of_incidence = self.calculate_angle_of_incidence(azimuth_sun, zenith_sun)
        direct_irradiance = np.maximum(dni * np.cos(np.radians(angle_of_incidence)), 0)
        diffuse_irradiance = dhi * (1 + np.cos(np.radians(self.tilt))) / 2
        ground_reflected_irradiance = (
            (dhi + dni * np.cos(np.radians(zenith_sun)))
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
        direct_irradiance, diffuse_irradiance, ground_reflected_irradiance = (
            self.calculate_gpoa_components(dhi, dni, azimuth_sun, zenith_sun)
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
            dhi, dni, azimuth_sun, zenith_sun
        )
        return diffuse / (direct + diffuse + ground_reflected)
