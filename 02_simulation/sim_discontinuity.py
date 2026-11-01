#!/usr/bin/env python3
"""CPW impedance discontinuity S11 simulation using MEEP 2D FDTD.

Two-step normalization approach:
1. Reference run: uniform 50Ω CPW → captures incident flux
2. DUT run: 50Ω → 35Ω step → measures reflected flux
S11 = reflected / incident

Also compares against analytical: Γ = (Z2 - Z1) / (Z2 + Z1)

MEEP units: mm, c=1. freq_GHz = freq_meep * 300.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp
from scipy.special import ellipk

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")
UM_TO_MM = 1e-3


def cpw_impedance(width_um, gap_um, eps_r=11.9):
    """Analytical CPW Z0."""
    W, G = width_um, gap_um
    k = W / (W + 2 * G)
    kp = np.sqrt(1.0 - k**2)
    eps_eff = (eps_r + 1.0) / 2.0
    Z0 = (30.0 * np.pi / np.sqrt(eps_eff)) * (ellipk(kp**2) / ellipk(k**2))
    return Z0, eps_eff


def run_sim(is_reference=False):
    """Run either reference (uniform) or DUT (step) simulation."""
    # Geometry params
    W1 = 10.0 * UM_TO_MM  # 50Ω section
    G1 = 6.0 * UM_TO_MM
    W2 = 20.0 * UM_TO_MM  # wider conductor, same gap → lower Z0
    G2 = 6.0 * UM_TO_MM   # keep same gap to break k-ratio symmetry

    L1 = 0.5  # mm, first section
    L2 = 0.5  # mm, second section
    total_L = L1 + L2

    eps_eff = 6.45

    pml_th = 0.08
    pad = 0.15
    sx = total_L + 2 * pad
    sy_half = max(W2 / 2 + G2, W1 / 2 + G1) + 0.1
    sy = 2 * sy_half
    cell = mp.Vector3(sx, sy)

    metal = mp.perfect_electric_conductor
    default_mat = mp.Medium(epsilon=eps_eff)

    if is_reference:
        # Uniform CPW (section 1 geometry throughout)
        geometry = [
            mp.Block(
                center=mp.Vector3(0, 0),
                size=mp.Vector3(total_L, W1),
                material=metal,
            ),
        ]
    else:
        # Step discontinuity
        x_mid = -total_L / 2 + L1
        geometry = [
            # Section 1
            mp.Block(
                center=mp.Vector3(-total_L / 2 + L1 / 2, 0),
                size=mp.Vector3(L1, W1),
                material=metal,
            ),
            # Section 2
            mp.Block(
                center=mp.Vector3(-total_L / 2 + L1 + L2 / 2, 0),
                size=mp.Vector3(L2, W2),
                material=metal,
            ),
        ]

    fcen = 0.04  # ~12 GHz
    df = 0.06    # wide band

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(-total_L / 2 + 0.05, 0),
            size=mp.Vector3(0, W1 * 3),
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(pml_th)],
        default_material=default_mat,
        resolution=40,
        dimensions=2,
    )

    nfreq = 60

    # Reflection monitor (between source and discontinuity)
    refl_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(-total_L / 2 + 0.12, 0),
            size=mp.Vector3(0, W1 * 3),
        ),
    )

    # Transmission monitor
    trans_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(
            center=mp.Vector3(total_L / 2 - 0.05, 0),
            size=mp.Vector3(0, max(W1, W2) * 3),
        ),
    )

    return sim, refl_mon, trans_mon, fcen, df, nfreq


def run():
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Step 1: Reference run
    print("=== Reference run (uniform CPW) ===")
    sim_ref, refl_ref, trans_ref, fcen, df, nfreq = run_sim(is_reference=True)
    sim_ref.run(until_after_sources=300)

    ref_refl_data = sim_ref.get_flux_data(refl_ref)
    ref_freqs = np.array(mp.get_flux_freqs(refl_ref))
    ref_trans_flux = np.array(mp.get_fluxes(trans_ref))
    sim_ref.reset_meep()

    # Step 2: DUT run
    print("\n=== DUT run (impedance step) ===")
    sim_dut, refl_dut, trans_dut, _, _, _ = run_sim(is_reference=False)

    # Load negative reference flux for S11 extraction
    sim_dut.load_minus_flux_data(refl_dut, ref_refl_data)

    sim_dut.run(until_after_sources=300)

    dut_freqs = np.array(mp.get_flux_freqs(refl_dut))
    dut_refl_flux = np.array(mp.get_fluxes(refl_dut))
    dut_trans_flux = np.array(mp.get_fluxes(trans_dut))

    # S11 = reflected / incident
    # reflected flux = -(flux with minus reference subtracted) when reflection exists
    # incident flux ≈ ref_trans_flux (transmitted in reference = incident)
    incident = ref_trans_flux
    incident_safe = np.where(incident > 0, incident, 1e-30)

    S11_linear = np.abs(dut_refl_flux) / incident_safe
    S11_db = 10 * np.log10(S11_linear + 1e-30)

    freq_ghz = dut_freqs * 300

    # Analytical reflection coefficient
    Z1, _ = cpw_impedance(10.0, 6.0)
    Z2, _ = cpw_impedance(20.0, 6.0)
    gamma_analytical = abs((Z2 - Z1) / (Z2 + Z1))
    S11_analytical_db = 20 * np.log10(gamma_analytical)

    print(f"\nAnalytical: Z1 = {Z1:.1f} Ω, Z2 = {Z2:.1f} Ω")
    print(f"Γ = {gamma_analytical:.4f} → S11 = {S11_analytical_db:.1f} dB")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(freq_ghz, S11_db, color="#2c3e50", linewidth=2, label="MEEP 2D FDTD")
    ax.axhline(
        S11_analytical_db,
        color="#e74c3c",
        linestyle="--",
        linewidth=2,
        label=f"Analytical Γ = {gamma_analytical:.3f} ({S11_analytical_db:.1f} dB)",
    )

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("S11 (dB)", fontsize=12)
    ax.set_title(
        f"Impedance Discontinuity S11 — CPW Step ({Z1:.0f}Ω → {Z2:.0f}Ω)\n"
        "2D FDTD (MEEP) vs Analytical",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 25)

    path = os.path.join(PLOT_DIR, "discontinuity_s11.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")

    sim_dut.reset_meep()


if __name__ == "__main__":
    run()
