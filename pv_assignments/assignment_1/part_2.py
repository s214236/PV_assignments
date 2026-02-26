"""Task 2 of assignment 1."""

import numpy as np

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

    def calculate_gpoa(
        self, dhi: float, dni: float, azimuth_sun: float, zenith_sun: float
    ) -> float:
        """Calculate the global plane of array irradiance.

        Args:
            dhi (float): Diffuse horizontal irradiance
            dni (float): Direct normal irradiance
            azimuth_sun (float): Azimuth angle of the sun
            zenith_sun (float): Zenith angle of the sun

        Returns:
            float: Global plane of array irradiance.
        """
        angle_of_incidence = np.degrees(
            np.arccos(
                np.cos(np.radians(zenith_sun)) * np.cos(np.radians(self.tilt))
                + np.sin(np.radians(zenith_sun))
                * np.sin(np.radians(self.tilt))
                * np.cos(np.radians(azimuth_sun - self.azimuth))
            )
        )
        direct_irradiance = np.maximum(dni * np.cos(np.radians(angle_of_incidence)), 0)
        diffuse_irradiance = dhi * (1 + np.cos(np.radians(self.tilt))) / 2
        ground_reflected_irradiance = (
            (dhi + direct_irradiance)
            * self.rho_g
            * (1 - np.cos(np.radians(self.tilt)))
            / 2
        )
        gpoa = direct_irradiance + diffuse_irradiance + ground_reflected_irradiance
        return gpoa


def main() -> None:
    """Main function for part 2 of the assignment."""
    print("\nPart 2-1")
    # NOTE: South convention is used. East=-90, West=90, North=+-180, South=0
    # Situation modelled: Morning sun, Panel facing towards evenings sun.
    azimuth_sun = -76
    zenith_sun = 63
    azimuth_panel = 63
    tilt_panel = 41

    angle_of_incidence = np.degrees(
        np.arccos(
            np.cos(np.radians(zenith_sun)) * np.cos(np.radians(tilt_panel))
            + np.sin(np.radians(zenith_sun))
            * np.sin(np.radians(tilt_panel))
            * np.cos(np.radians(azimuth_sun - azimuth_panel))
        )
    )
    print(f"Angle of incidence: {angle_of_incidence}")

    dni = 600
    dhi = 120
    rho_g = 0.2
    direct_irradiance = np.maximum(dni * np.cos(np.radians(angle_of_incidence)), 0)
    diffuse_irradiance = dhi * (1 + np.cos(np.radians(tilt_panel))) / 2
    ground_reflected_irradiance = (
        (dhi + direct_irradiance) * rho_g * (1 - np.cos(np.radians(tilt_panel))) / 2
    )
    print(f"Direct irradiance: {direct_irradiance}")
    print(f"Diffuse irradiance: {diffuse_irradiance}")
    print(f"Ground reflected irradiance: {ground_reflected_irradiance}")

    print("\nPart 2-2")

    df = load_data()
    print(df.head())


if __name__ == "__main__":
    main()
