#!/usr/bin/env python3
"""Coupled CPW crosstalk simulation using MEEP 2D FDTD.

Simulates two parallel CPW lines in an effective-epsilon medium.
Measures NEXT (near-end) and FEXT (far-end) crosstalk vs frequency
for three separation distances.

MEEP units: length in mm, c=1, so freq 0.01 → 3 GHz, freq 0.1 → 30 GHz.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meep as mp

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")

# Simulation parameters (all in mm)
UM_TO_MM = 1e-3  # convert um to mm


def build_coupled_cpw_sim(
    separation_um=20.0,
    width_um=10.0,
    gap_um=6.0,
    coupling_length_um=500.0,
    eps_eff=6.45,
    resolution=40,
):
    """Build MEEP simulation for coupled CPW crosstalk.

    Uses a 2D effective-medium approach: PEC metal strips in a uniform
    dielectric with eps_eff = (eps_r + 1)/2.

    Returns: simulation, flux monitors dict
    """
    W = width_um * UM_TO_MM
    G = gap_um * UM_TO_MM
    S = separation_um * UM_TO_MM
    L = coupling_length_um * UM_TO_MM

    # Y-positions of the two conductor centers
    y_agg = (S / 2 + G + W / 2)
    y_vic = -y_agg

    # Computational cell
    pad_x = 0.2  # mm padding
    pad_y = 0.15
    sx = L + 2 * pad_x
    sy = 2 * (abs(y_agg) + W / 2 + G + 0.05) + 2 * pad_y
    cell = mp.Vector3(sx, sy)

    # PML boundary
    pml = [mp.PML(thickness=0.05)]

    # Background: effective dielectric
    default_material = mp.Medium(epsilon=eps_eff)

    # Metal strips as perfect conductors (metal = high conductivity)
    metal = mp.perfect_electric_conductor

    # Geometry: aggressor conductor, victim conductor
    # (Ground planes omitted for 2D simplicity — they're effectively at infinity)
    geometry = [
        # Aggressor
        mp.Block(
            center=mp.Vector3(0, y_agg),
            size=mp.Vector3(L, W),
            material=metal,
        ),
        # Victim
        mp.Block(
            center=mp.Vector3(0, y_vic),
            size=mp.Vector3(L, W),
            material=metal,
        ),
    ]

    # Source: Gaussian pulse on aggressor input (left side)
    fcen = 0.03  # 9 GHz center
    df = 0.04    # wide bandwidth (3-21 GHz)
    src_x = -L / 2 + 0.02

    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Ez,
            center=mp.Vector3(src_x, y_agg),
            size=mp.Vector3(0, W * 1.5),
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml,
        default_material=default_material,
        resolution=resolution,
        dimensions=2,
    )

    # Flux monitors
    nfreq = 50
    flux_x_out = L / 2 - 0.02
    flux_x_in = -L / 2 + 0.02
    monitor_width = W * 2

    # Aggressor output (transmitted)
    agg_out = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(flux_x_out, y_agg), size=mp.Vector3(0, monitor_width)),
    )
    # Aggressor input (reference)
    agg_in = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(flux_x_in, y_agg), size=mp.Vector3(0, monitor_width)),
    )
    # NEXT: near-end crosstalk (victim, same side as source)
    next_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(flux_x_in, y_vic), size=mp.Vector3(0, monitor_width)),
    )
    # FEXT: far-end crosstalk (victim, opposite side)
    fext_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(flux_x_out, y_vic), size=mp.Vector3(0, monitor_width)),
    )

    return sim, {"agg_out": agg_out, "agg_in": agg_in, "next": next_mon, "fext": fext_mon}


def run():
    os.makedirs(PLOT_DIR, exist_ok=True)

    separations = [10.0, 20.0, 50.0]
    results = {}

    for sep in separations:
        print(f"\n--- Simulating coupled CPW, separation = {sep} um ---")
        sim, monitors = build_coupled_cpw_sim(separation_um=sep, resolution=30)
        sim.run(until_after_sources=200)

        freqs = np.array(mp.get_flux_freqs(monitors["agg_in"]))
        p_in = np.array(mp.get_fluxes(monitors["agg_in"]))
        p_next = np.abs(np.array(mp.get_fluxes(monitors["next"])))
        p_fext = np.abs(np.array(mp.get_fluxes(monitors["fext"])))

        # Avoid division by zero
        p_in_safe = np.where(p_in > 0, p_in, 1e-30)

        next_db = 10 * np.log10(p_next / p_in_safe + 1e-30)
        fext_db = 10 * np.log10(p_fext / p_in_safe + 1e-30)

        # Convert MEEP freq to GHz (freq * c / mm = freq * 3e11 / 1e-3 = freq * 3e8 ... )
        # In MEEP with mm units: f_GHz = f_meep * 300
        freq_ghz = freqs * 300.0

        results[sep] = {"freq_ghz": freq_ghz, "next_db": next_db, "fext_db": fext_db}

        sim.reset_meep()

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, sep in enumerate(separations):
        r = results[sep]
        ax1.plot(r["freq_ghz"], r["next_db"], color=colors[i], linewidth=2,
                 label=f"S = {sep} μm")
        ax2.plot(r["freq_ghz"], r["fext_db"], color=colors[i], linewidth=2,
                 label=f"S = {sep} μm")

    for ax, title in [(ax1, "NEXT (Near-End Crosstalk)"), (ax2, "FEXT (Far-End Crosstalk)")]:
        ax.set_xlabel("Frequency (GHz)", fontsize=12)
        ax.set_ylabel("Crosstalk (dB)", fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 25)

    fig.suptitle("Coupled CPW Crosstalk — 2D FDTD (MEEP)", fontsize=14, y=1.02)
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "coupled_cpw_next_fext.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    run()
