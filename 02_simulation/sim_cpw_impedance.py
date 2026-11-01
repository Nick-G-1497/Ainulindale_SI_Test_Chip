#!/usr/bin/env python3
"""Analytical CPW impedance vs geometry using Wen 1969 conformal mapping.

Uses the closed-form expression for CPW characteristic impedance:
    Z0 = (30*pi / sqrt(eps_eff)) * (K(k') / K(k))
where:
    k  = W / (W + 2*G)          (geometry ratio)
    k' = sqrt(1 - k^2)          (complementary modulus)
    K  = complete elliptic integral of the first kind
    eps_eff = (eps_r + 1) / 2   (quasi-static approximation for CPW on dielectric half-space)

Reference: C.P. Wen, "Coplanar Waveguide: A Surface Strip Transmission Line
Suitable for Nonreciprocal Gyromagnetic Device Applications," IEEE Trans. MTT, 1969.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import ellipk

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")


def cpw_impedance(width_um, gap_um, eps_r=11.9):
    """Calculate CPW characteristic impedance (ohms).

    Args:
        width_um: Center conductor width (um).
        gap_um: Conductor-ground gap (um).
        eps_r: Substrate relative permittivity (11.9 for Si).

    Returns:
        Z0 in ohms, eps_eff.
    """
    W = width_um
    G = gap_um
    k = W / (W + 2 * G)
    kp = np.sqrt(1.0 - k**2)

    eps_eff = (eps_r + 1.0) / 2.0

    Z0 = (30.0 * np.pi / np.sqrt(eps_eff)) * (ellipk(kp**2) / ellipk(k**2))
    return Z0, eps_eff


def run():
    os.makedirs(PLOT_DIR, exist_ok=True)

    widths = np.linspace(3, 30, 100)
    gaps = [3.0, 6.0, 12.0]

    fig, ax = plt.subplots(figsize=(10, 7))

    for gap in gaps:
        Z0_vals = []
        for w in widths:
            z0, _ = cpw_impedance(w, gap)
            Z0_vals.append(z0)
        ax.plot(widths, Z0_vals, linewidth=2, label=f"G = {gap} μm")

    # Reference lines
    ax.axhline(50, color="k", linestyle="--", linewidth=1, alpha=0.7, label="50 Ω")

    # Mark the three design points
    design_points = [
        (5, 3, "High-Z"),
        (10, 6, "50Ω design"),
        (20, 12, "Low-Z"),
    ]
    for w, g, label in design_points:
        z0, _ = cpw_impedance(w, g)
        ax.plot(w, z0, "ko", markersize=8)
        ax.annotate(
            f"{label}\nW={w}, G={g}\nZ₀={z0:.1f} Ω",
            xy=(w, z0),
            xytext=(w + 2, z0 + 8),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )

    ax.set_xlabel("Center Conductor Width W (μm)", fontsize=12)
    ax.set_ylabel("Characteristic Impedance Z₀ (Ω)", fontsize=12)
    ax.set_title(
        "CPW Impedance vs Geometry (Si substrate, εᵣ=11.9)\n"
        "Wen 1969 Conformal Mapping",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(3, 30)
    ax.set_ylim(0, 120)

    path = os.path.join(PLOT_DIR, "cpw_impedance_vs_geometry.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print summary table
    print("\nDesign Point Summary:")
    print(f"{'W(um)':>8} {'G(um)':>8} {'Z0(Ω)':>10} {'eps_eff':>10}")
    print("-" * 40)
    for w, g, label in design_points:
        z0, eps_eff = cpw_impedance(w, g)
        print(f"{w:>8.1f} {g:>8.1f} {z0:>10.2f} {eps_eff:>10.2f}")


if __name__ == "__main__":
    run()
