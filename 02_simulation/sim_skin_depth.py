#!/usr/bin/env python3
"""Skin depth comparison: 300K (room temperature) vs 4K (cryogenic).

This simulation demonstrates how dramatically the skin depth changes
at cryogenic temperatures, directly relevant to cryo-CMOS interconnects.

Physics:
    delta = sqrt(2 * rho / (omega * mu_0))

    For copper at 300K:  rho ≈ 1.68e-8 Ω·m  (bulk)
    For copper at 4K:    rho ≈ 1.5e-10 Ω·m   (RRR~100, typical for high-purity Cu)
    For aluminum at 300K: rho ≈ 2.65e-8 Ω·m
    For aluminum at 4K:   superconducting (Tc=1.2K), but normal state rho ≈ 2.5e-10 Ω·m

    The key insight for cryo-CMOS:
    - At 300K, skin depth at 10 GHz is ~660 nm (Cu) — comparable to thin-film thickness
    - At 4K, skin depth at 10 GHz is ~62 nm (Cu) — much smaller than film thickness
    - This means surface roughness and film quality matter MORE at cryo temperatures
    - For superconducting NbTi (Tc~10K), the London penetration depth replaces skin depth

Also includes a MEEP 2D simulation of field penetration into a lossy conductor
at both temperatures to visually demonstrate the difference.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")

MU_0 = 4 * np.pi * 1e-7  # H/m


def skin_depth(rho, freq):
    """Calculate skin depth (m).

    Args:
        rho: Resistivity (Ω·m)
        freq: Frequency (Hz)
    Returns:
        Skin depth δ (m)
    """
    omega = 2 * np.pi * freq
    return np.sqrt(2 * rho / (omega * MU_0))


def run_analytical():
    """Generate analytical skin depth plots."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    freqs_hz = np.logspace(8, 11, 200)  # 100 MHz to 100 GHz
    freqs_ghz = freqs_hz / 1e9

    # Material properties
    materials = {
        "Cu (300K)": {"rho": 1.68e-8, "color": "#e74c3c", "ls": "-"},
        "Cu (4K, RRR=100)": {"rho": 1.5e-10, "color": "#e74c3c", "ls": "--"},
        "Al (300K)": {"rho": 2.65e-8, "color": "#3498db", "ls": "-"},
        "Al (4K, normal)": {"rho": 2.5e-10, "color": "#3498db", "ls": "--"},
        "NbTi (300K)": {"rho": 8.0e-7, "color": "#2ecc71", "ls": "-"},
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Skin depth vs frequency
    for name, props in materials.items():
        delta = skin_depth(props["rho"], freqs_hz) * 1e6  # convert to μm
        ax1.loglog(freqs_ghz, delta, color=props["color"], linestyle=props["ls"],
                   linewidth=2.5, label=name)

    # Reference lines for typical film thicknesses
    for thickness, label in [(1.0, "1 μm film"), (0.1, "100 nm film")]:
        ax1.axhline(thickness, color="gray", linestyle=":", alpha=0.5)
        ax1.text(0.12, thickness * 1.1, label, fontsize=9, color="gray")

    ax1.set_xlabel("Frequency (GHz)", fontsize=12)
    ax1.set_ylabel("Skin Depth δ (μm)", fontsize=12)
    ax1.set_title("Skin Depth vs Frequency", fontsize=13)
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.3, which="both")
    ax1.set_xlim(0.1, 100)
    ax1.set_ylim(0.01, 100)

    # Plot 2: Skin depth ratio (300K / 4K) for Cu
    rho_300K = 1.68e-8
    rho_4K = 1.5e-10
    delta_300K = skin_depth(rho_300K, freqs_hz) * 1e6
    delta_4K = skin_depth(rho_4K, freqs_hz) * 1e6
    ratio = delta_300K / delta_4K

    ax2.semilogx(freqs_ghz, ratio, color="#9b59b6", linewidth=2.5)
    ax2.axhline(np.sqrt(rho_300K / rho_4K), color="gray", linestyle="--",
                label=f"√(ρ₃₀₀/ρ₄) = {np.sqrt(rho_300K/rho_4K):.1f}×")
    ax2.fill_between(freqs_ghz, 1, ratio, alpha=0.1, color="#9b59b6")

    ax2.set_xlabel("Frequency (GHz)", fontsize=12)
    ax2.set_ylabel("Skin Depth Ratio (300K / 4K)", fontsize=12)
    ax2.set_title("Copper: Cryogenic Skin Depth Reduction", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, which="both")
    ax2.set_xlim(0.1, 100)

    fig.suptitle(
        "Skin Depth at Room Temperature vs Cryogenic — Implications for Cryo-CMOS Interconnects",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "skin_depth_comparison.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # Print summary table
    print("\nSkin Depth Summary at Key Frequencies:")
    print(f"{'Material':<20} {'1 GHz':>12} {'10 GHz':>12} {'40 GHz':>12}")
    print("-" * 60)
    for name, props in materials.items():
        d1 = skin_depth(props["rho"], 1e9) * 1e6
        d10 = skin_depth(props["rho"], 10e9) * 1e6
        d40 = skin_depth(props["rho"], 40e9) * 1e6
        print(f"{name:<20} {d1:>10.3f} μm {d10:>10.3f} μm {d40:>10.3f} μm")


def run_meep_penetration():
    """MEEP 2D simulation of field penetration at 300K vs 4K.

    Models a plane wave incident on a lossy conductor slab to visualize
    the exponential decay (skin effect).
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("\nRunning MEEP field penetration simulation...")

    freq_ghz = 10.0
    # In MEEP mm units: freq_meep = freq_ghz / 300
    freq_meep = freq_ghz / 300.0

    # We need to map physical conductivity to MEEP units.
    # MEEP conductivity (in Meep units with mm): σ_meep = σ_physical / (2π × freq × ε₀)
    # But easier: use the Drude model for metals.
    # For a good conductor, ε ≈ 1 + iσ/(ωε₀).
    # In MEEP, we can use Medium(D_conductivity=σ) where σ is in MEEP units.
    #
    # Skin depth in MEEP: δ_meep = δ_physical / (unit_length)
    # For Cu at 300K, 10 GHz: δ = 0.66 μm = 6.6e-4 mm
    # This is WAY too small to resolve in MEEP (need resolution >> 1/δ).
    #
    # Instead, use scaled conductivities to demonstrate the concept with
    # resolvable skin depths.

    # Use artificially low conductivities for visualization
    # Scale so skin depth is ~0.1-1 mm (resolvable at resolution 100)
    # δ = sqrt(2/(ω μ σ)) → σ = 2/(ω μ δ²)
    # For δ = 0.2 mm, f_meep = 0.0333:
    # ω = 2π × 0.0333 = 0.2094
    # σ_meep = 2 / (0.2094 × μ₀_meep × 0.2²)
    # In MEEP, μ₀ = 1 (c=1 units), so:
    # σ_meep = 2 / (0.2094 × 0.04) = 238.7

    delta_300K_target = 0.3  # mm (scaled)
    delta_4K_target = 0.03   # mm (scaled, 10× smaller)

    omega = 2 * np.pi * freq_meep
    # In MEEP: σ_D acts as D_conductivity, giving σ_eff = σ_D * ω
    # Actually MEEP's D_conductivity gives damping: ε → ε + iσ_D/ω
    # So: skin depth δ = sqrt(2/(ω × σ_D)) when σ_D >> ωε

    # More directly: use meep Medium(D_conductivity=val)
    # The D_conductivity in MEEP gives: d(D)/dt term → loss
    # Effective σ_eff = ε₀ × σ_D (in Meep units, ε₀=1)
    # δ = 1 / sqrt(π f μ σ_eff) = 1 / sqrt(π f σ_D)

    sigma_300K = 1.0 / (np.pi * freq_meep * delta_300K_target**2)
    sigma_4K = 1.0 / (np.pi * freq_meep * delta_4K_target**2)

    configs = [
        ("300K", sigma_300K, delta_300K_target, "#e74c3c"),
        ("4K", sigma_4K, delta_4K_target, "#3498db"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (label, sigma, delta_target, color) in enumerate(configs):
        # Simulation domain
        slab_thickness = 1.5  # mm
        pml_th = 0.3
        pad = 0.5
        sx = slab_thickness + 2 * pad + 2 * pml_th
        cell = mp.Vector3(sx, 0, 0)

        # Lossy conductor slab on the right side
        slab_x = pad  # slab starts at x = pad from center
        geometry = [
            mp.Block(
                center=mp.Vector3(slab_x + slab_thickness / 2, 0),
                size=mp.Vector3(slab_thickness, mp.inf),
                material=mp.Medium(epsilon=1, D_conductivity=sigma),
            ),
        ]

        sources = [
            mp.Source(
                src=mp.ContinuousSource(frequency=freq_meep),
                component=mp.Ex,  # 1D MEEP supports Ex, Hy (or Ey, Hx)
                center=mp.Vector3(-sx / 2 + pml_th + 0.1, 0),
            ),
        ]

        sim = mp.Simulation(
            cell_size=cell,
            geometry=geometry,
            sources=sources,
            boundary_layers=[mp.PML(pml_th)],
            resolution=200,
            dimensions=1,
        )

        # Run long enough for steady state
        sim.run(until=40 / freq_meep)

        # Get field profile
        ez = sim.get_array(center=mp.Vector3(), size=mp.Vector3(sx),
                           component=mp.Ex)
        x_coords = np.linspace(-sx/2, sx/2, len(ez))

        # Normalize
        ez_norm = np.abs(ez) / (np.max(np.abs(ez)) + 1e-30)

        ax = axes[idx]
        ax.plot(x_coords * 1000, ez_norm, color=color, linewidth=2)
        ax.axvline(slab_x * 1000, color="gray", linestyle="--", alpha=0.7,
                   label="Metal surface")
        ax.axvspan(slab_x * 1000, (slab_x + slab_thickness) * 1000,
                   alpha=0.1, color="gray")

        # Mark skin depth
        ax.axvline((slab_x + delta_target) * 1000, color=color,
                   linestyle=":", linewidth=2,
                   label=f"δ = {delta_target*1000:.0f} μm (scaled)")

        ax.set_xlabel("Position (μm)", fontsize=12)
        ax.set_ylabel("|Ex| (normalized)", fontsize=12)
        ax.set_title(f"Field Penetration at {label}\n(scaled δ = {delta_target*1000:.0f} μm)",
                     fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.1)

        sim.reset_meep()

    fig.suptitle(
        f"Skin Effect: Field Penetration into Conductor — {freq_ghz:.0f} GHz\n"
        "MEEP 1D FDTD (scaled conductivities for visualization)",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "skin_depth_meep_penetration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run_analytical_penetration():
    """Generate an analytical plot showing exponential field decay into conductor."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    freq = 10e9  # 10 GHz
    x_um = np.linspace(0, 5, 500)  # depth in μm

    materials = {
        "Cu (300K)": {"rho": 1.68e-8, "color": "#e74c3c"},
        "Cu (4K)": {"rho": 1.5e-10, "color": "#3498db"},
        "Al (300K)": {"rho": 2.65e-8, "color": "#e67e22"},
        "Al (4K)": {"rho": 2.5e-10, "color": "#2ecc71"},
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, props in materials.items():
        delta = skin_depth(props["rho"], freq) * 1e6  # μm
        decay = np.exp(-x_um / delta)
        ax.plot(x_um, decay, color=props["color"], linewidth=2.5, label=f"{name} (δ = {delta:.3f} μm)")

    # Mark typical film thicknesses
    for t, label in [(0.5, "500 nm film"), (1.0, "1 μm film")]:
        ax.axvline(t, color="gray", linestyle=":", alpha=0.5)
        ax.text(t + 0.05, 0.95, label, fontsize=9, color="gray", rotation=90, va="top")

    ax.set_xlabel("Depth into Conductor (μm)", fontsize=12)
    ax.set_ylabel("Normalized Field Amplitude |E/E₀|", fontsize=12)
    ax.set_title(
        "Field Penetration (Skin Effect) at 10 GHz — 300K vs 4K\n"
        "Exponential decay: E(x) = E₀ exp(-x/δ)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 1.05)

    path = os.path.join(PLOT_DIR, "skin_depth_penetration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run():
    run_analytical()
    run_analytical_penetration()


if __name__ == "__main__":
    run()
