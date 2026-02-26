"""Task 1 of assignment 1."""

from importlib import resources
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pv_assignments.assignment_1.data.part_1_loader import load_data

def _economist_style(ax: plt.Axes) -> None:
    """Apply an Economist-like style to an axes."""
    # Subtle horizontal grid only
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.grid(axis="x", visible=False)

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Slightly soften remaining spines
    ax.spines["left"].set_alpha(0.8)
    ax.spines["bottom"].set_alpha(0.8)

    # Ticks
    ax.tick_params(axis="both", which="major", labelsize=10)

PV_colors = {
    "lines": [
        "#0B1F3B",  # deep navy (anchor)
        "#E69F00",  # solar orange
        "#56B4E9",  # sky blue
        "#009E73",  # teal/green
        "#D55E00",  # heat / vermillion
        "#F0C66D",  # amber highlight
        "#4E5A61",  # slate grey (extra)
    ],
    "grid": "#B7B7B7",
    "text": "#222222",
}

def _figure_path(fig_title: str) -> str:
    # This points to pv_assignments/assignment_1/figures/
    p = resources.files("pv_assignments.assignment_1.figures").joinpath(f"{fig_title}.png")
    return str(p)

def plot(
    dfs: list[pd.DataFrame],
    labels: list[str],
    value_names: list[str],
    peak_wavelengths: bool = False,
    title: str | None = None,
    fig_title: str | None = None,
) -> None:
    """Plot spectral irradiance for multiple dataframes.

    Args:
        dfs (list[pd.DataFrame]): DataFrames containing the spectral irradiance data.
        labels (list[str]): Labels for each DataFrame.
        value_names (list[str]): Names of the columns to plot.
        peak_wavelengths (bool, optional): Whether to mark peak wavelengths with vertical lines.
            Defaults to False.
    """

    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    fig_width = 7
    fig, ax = plt.subplots(figsize=(fig_width, fig_width / phi))

    palette = PV_colors["lines"]
    i = 0

    # Plot series
    for df, label in zip(dfs, labels, strict=True):
        x = df["Wavelength (nm)"].to_numpy()
        for value_name in value_names:
            series_label = f"{label} - {value_name}"
            y = df[value_name].to_numpy()

            ax.plot(
                x, y,
                color = palette[i % len(palette)],
                linewidth=1.3,
                label=series_label,
            )
            i+=1

            if peak_wavelengths:
                peak = peak_wavelength([df], [label], [value_name], silent=True)[series_label]
                ax.axvline(
                    x=peak["wavelength"],
                    color="black",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1.2,
                )

    # Labels & title
    ax.set_xlabel("Wavelength (nm)", fontsize=12, labelpad=10)
    ax.set_ylabel("Global to perpendicular plane (W/m²/nm)", fontsize=12, labelpad=10)

    # Put title in the *figure* header (not in axes)
    fig.suptitle(
        title,
        x=0.11, y=0.98,  # left-ish, high up
        ha="left",
        fontsize=14,
        fontweight="bold",
    )

    _economist_style(ax)

    # Legend also in *figure* header (not axes)
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.93),
        ncol=1,
        fontsize=8,
        handlelength=2.5,
        columnspacing=1.2,
    )

    # Reserve top space for title + legend
    fig.tight_layout()

    # --- save ---
    if fig_title is None:
        fig_title = title

    save_path = _figure_path(fig_title)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure:", save_path)

# Not used
def plot_peter(
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
    plot(dfs, labels, value_names, peak_wavelengths=True, title="Spectral irradiance for different air masses", fig_title="Part 1-1")

    print("\nPart 1-2")
    dfs = [load_data("1.5")]
    labels = ["AM 1.5"]
    value_names = [
        "Direct to horizontal plane (W/m2/nm)",
        "Diffuse to horizontal plane (W/m2/nm)",
        "Global to horizontal plane  (W/m2/nm)",
    ]
    plot(dfs=dfs, labels=labels, value_names=value_names, title="Spectral irradiance - Horizontal plane", fig_title="Part 1-2")
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
    plot(dfs=dfs, labels=labels, value_names=value_names, title="Spectral irradiance for different water vapor contents", fig_title="Part 1-3")


if __name__ == "__main__":
    main()
