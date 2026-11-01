#!/usr/bin/env python3
"""Generate GDS layout and annotated PNG for the SI test chip."""

import sys
import os
import matplotlib
matplotlib.use("Agg")

# Add layout dir to path so local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gdsfactory as gf
gf.gpdk.PDK.activate()
import matplotlib.pyplot as plt

from test_chip import si_test_chip
from cpw_cells import cpw_line_set
from trl_cells import trl_cal_set
from coupled_cpw_cells import coupled_cpw_set
from resonator_cells import cpw_quarter_wave_resonator
from discontinuity_cells import cpw_impedance_step

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")


def generate_gds():
    """Generate the full test chip GDS."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    chip = si_test_chip()
    gds_path = os.path.join(OUTPUT_DIR, "si_testchip.gds")
    chip.write_gds(gds_path)
    print(f"GDS written to {gds_path}")
    return chip


def generate_annotated_png(chip):
    """Generate annotated matplotlib rendering of the chip."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 14))
    chip.plot(ax=ax)
    ax.set_title("SI Test Chip — Signal Integrity Characterization Structures", fontsize=14)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_aspect("equal")

    png_path = os.path.join(OUTPUT_DIR, "si_testchip_annotated.png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Annotated PNG written to {png_path}")


def generate_individual_pngs():
    """Generate individual PNGs of each test structure."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    structures = {
        "cpw_line_set": cpw_line_set(),
        "trl_cal_set": trl_cal_set(),
        "coupled_cpw_set": coupled_cpw_set(),
        "resonator": cpw_quarter_wave_resonator(),
        "impedance_step": cpw_impedance_step(),
    }

    for name, comp in structures.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        comp.plot(ax=ax)
        ax.set_title(name.replace("_", " ").title())
        ax.set_aspect("equal")
        path = os.path.join(PLOT_DIR, f"{name}_layout.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {path}")


if __name__ == "__main__":
    print("=== Generating SI Test Chip Layout ===")
    chip = generate_gds()
    print("\nGenerating annotated PNG...")
    generate_annotated_png(chip)
    print("\nGenerating individual structure PNGs...")
    generate_individual_pngs()
    print("\nDone!")
