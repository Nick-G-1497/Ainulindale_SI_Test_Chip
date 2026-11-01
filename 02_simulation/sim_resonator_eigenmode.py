#!/usr/bin/env python3
"""Quarter-wave CPW resonator eigenmode simulation — THE CENTERPIECE.

Two-part simulation:
  Part A: Closed cavity eigenmode analysis — demonstrates Harminv mode
          extraction with clean Q-factors (isolated resonator)
  Part B: Coupled resonator with feedline — realistic measurement config
          showing how coupling affects the spectrum

2D Model: Parallel-plate channel (PEC walls + dielectric fill) using Hz
polarization (TE modes). Hz/Ex/Ey fields support the TEM-like mode.

Physics:
    Closed cavity half-wave modes:  f_n = n * c / (2 * L * sqrt(eps_eff))
    Quarter-wave stub modes:        f_n = (2n-1) * c / (4 * L * sqrt(eps_eff))

MEEP units: mm, c=1. freq_GHz = freq_meep * 300.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import meep as mp

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")


def run_closed_cavity():
    """Part A: Closed rectangular PEC cavity — clean eigenmode extraction."""
    print("\n" + "=" * 50)
    print("Part A: Closed Cavity Eigenmode Analysis")
    print("=" * 50)

    # Cavity dimensions
    cav_w = 0.2     # width (mm) — 200 um
    cav_l = 2.0     # length (mm) — 2000 um
    wall_t = 0.04   # wall thickness
    eps_eff = 6.45

    # Expected half-wave modes: f_n = n / (2 * L * sqrt(eps))
    f1_exp = 1.0 / (2 * cav_l * np.sqrt(eps_eff))
    print(f"Expected modes: f1 = {f1_exp*300:.1f} GHz, f2 = {2*f1_exp*300:.1f} GHz, f3 = {3*f1_exp*300:.1f} GHz")

    # Cell
    pml_th = 0.15
    pad = 0.2
    sx = cav_w + 2 * wall_t + 2 * pad + 2 * pml_th
    sy = cav_l + 2 * wall_t + 2 * pad + 2 * pml_th
    cell = mp.Vector3(sx, sy)

    metal = mp.perfect_electric_conductor
    default_mat = mp.Medium(epsilon=eps_eff)

    # Fully enclosed PEC cavity
    geometry = [
        # Left wall
        mp.Block(center=mp.Vector3(-cav_w/2 - wall_t/2, 0),
                 size=mp.Vector3(wall_t, cav_l + 2*wall_t), material=metal),
        # Right wall
        mp.Block(center=mp.Vector3(+cav_w/2 + wall_t/2, 0),
                 size=mp.Vector3(wall_t, cav_l + 2*wall_t), material=metal),
        # Bottom wall
        mp.Block(center=mp.Vector3(0, -cav_l/2 - wall_t/2),
                 size=mp.Vector3(cav_w + 2*wall_t, wall_t), material=metal),
        # Top wall
        mp.Block(center=mp.Vector3(0, +cav_l/2 + wall_t/2),
                 size=mp.Vector3(cav_w + 2*wall_t, wall_t), material=metal),
    ]

    # Broadband source off-center to excite multiple modes
    fcen = 0.06   # ~18 GHz
    df = 0.08     # wide band
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,
            center=mp.Vector3(0, -cav_l * 0.23),  # off-center
            size=mp.Vector3(cav_w * 0.5, 0),
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

    harminv_pt = mp.Vector3(0, -cav_l * 0.15)
    harminv = mp.Harminv(mp.Hz, harminv_pt, fcen, df)

    hz_data = []
    def record_field(sim):
        hz_data.append(sim.get_field_point(mp.Hz, harminv_pt))

    print("Running closed cavity eigenmode simulation...")
    sim.run(
        mp.at_every(0.5, record_field),
        mp.after_sources(harminv),
        until_after_sources=3000,
    )

    # Extract Harminv results
    print("\n=== Harminv Results (Closed Cavity) ===")
    modes = harminv.modes
    valid_modes = []
    if modes:
        for i, mode in enumerate(modes):
            f_ghz = mode.freq * 300
            Q = mode.Q
            amp = abs(mode.amp)
            if Q > 5 and 1 < f_ghz < 60:
                valid_modes.append(mode)
                print(f"  Mode {i}: f = {f_ghz:.2f} GHz, Q = {Q:.0f}, |amp| = {amp:.6f}")
    if not valid_modes:
        print("  No modes found by Harminv, using FFT peaks")

    # FFT spectrum
    hz_arr = np.array(hz_data, dtype=complex)
    dt = 0.5
    N = len(hz_arr)
    window = np.hanning(N)
    hz_windowed = np.real(hz_arr) * window
    spectrum = np.abs(np.fft.rfft(hz_windowed))
    freqs_meep = np.fft.rfftfreq(N, d=dt)
    freqs_ghz = freqs_meep * 300

    # FFT peak detection as fallback
    if not valid_modes:
        from scipy.signal import find_peaks
        spectrum_norm = spectrum / (spectrum.max() + 1e-30)
        peaks, _ = find_peaks(spectrum_norm, height=0.05, distance=3, prominence=0.03)
        print("  FFT peaks:")
        for p in peaks:
            if 1 < freqs_ghz[p] < 60:
                print(f"    f = {freqs_ghz[p]:.1f} GHz")

    # Get field patterns at simulation end
    hz_field = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)

    sim.reset_meep()
    return {
        "valid_modes": valid_modes,
        "freqs_ghz": freqs_ghz,
        "spectrum": spectrum,
        "hz_field": hz_field,
        "cell": (sx, sy),
        "cav_w": cav_w,
        "cav_l": cav_l,
        "wall_t": wall_t,
        "f1_exp": f1_exp,
        "eps_eff": eps_eff,
    }


def run_coupled_resonator():
    """Part B: Quarter-wave resonator coupled to feedline."""
    print("\n" + "=" * 50)
    print("Part B: Coupled Resonator with Feedline")
    print("=" * 50)

    channel_w = 0.2
    res_len = 2.0
    feed_len = 3.0
    wall_t = 0.04
    coupling_w = 0.06  # narrow coupling for moderate Q
    eps_eff = 6.45

    f_qw = 1.0 / (4 * res_len * np.sqrt(eps_eff))
    print(f"Expected quarter-wave fundamental: {f_qw*300:.1f} GHz")

    pml_th = 0.2
    pad = 0.3
    sx = feed_len + 2 * pad + 2 * pml_th
    sy = res_len + channel_w + pad + 2 * pml_th + 0.5
    cell = mp.Vector3(sx, sy)

    y_feed = -sy / 2 + pml_th + pad + channel_w / 2
    y_res_start = y_feed + channel_w / 2 + channel_w / 2
    y_res_end = y_res_start + res_len

    metal = mp.perfect_electric_conductor
    default_mat = mp.Medium(epsilon=eps_eff)

    geometry = [
        # Feedline bottom wall
        mp.Block(center=mp.Vector3(0, y_feed - channel_w/2 - wall_t/2),
                 size=mp.Vector3(feed_len + 0.4, wall_t), material=metal),
        # Feedline top wall — left section
        mp.Block(center=mp.Vector3(-(coupling_w/2 + (feed_len/2 - coupling_w/2)/2),
                                    y_feed + channel_w/2 + wall_t/2),
                 size=mp.Vector3(feed_len/2 - coupling_w/2, wall_t), material=metal),
        # Feedline top wall — right section
        mp.Block(center=mp.Vector3(+(coupling_w/2 + (feed_len/2 - coupling_w/2)/2),
                                    y_feed + channel_w/2 + wall_t/2),
                 size=mp.Vector3(feed_len/2 - coupling_w/2, wall_t), material=metal),
        # Resonator left wall
        mp.Block(center=mp.Vector3(-channel_w/2 - wall_t/2, (y_res_start + y_res_end)/2),
                 size=mp.Vector3(wall_t, res_len + wall_t), material=metal),
        # Resonator right wall
        mp.Block(center=mp.Vector3(+channel_w/2 + wall_t/2, (y_res_start + y_res_end)/2),
                 size=mp.Vector3(wall_t, res_len + wall_t), material=metal),
        # Shorted end (top)
        mp.Block(center=mp.Vector3(0, y_res_end + wall_t/2),
                 size=mp.Vector3(channel_w + 2*wall_t, wall_t), material=metal),
    ]

    fcen = 0.05
    df = 0.06
    sources = [
        mp.Source(
            src=mp.GaussianSource(fcen, fwidth=df),
            component=mp.Hz,
            center=mp.Vector3(-feed_len/2 + 0.2, y_feed),
            size=mp.Vector3(0, channel_w * 0.8),
        ),
    ]

    sim = mp.Simulation(
        cell_size=cell,
        geometry=geometry,
        sources=sources,
        boundary_layers=[mp.PML(pml_th)],
        default_material=default_mat,
        resolution=30,
        dimensions=2,
    )

    # Transmission flux monitors
    nfreq = 80
    trans_mon = sim.add_flux(
        fcen, df, nfreq,
        mp.FluxRegion(center=mp.Vector3(feed_len/2 - 0.15, y_feed),
                      size=mp.Vector3(0, channel_w)),
    )

    # Also record in the resonator for FFT
    res_pt = mp.Vector3(0, y_res_start + res_len * 0.4)
    hz_data = []
    def record_field(sim):
        hz_data.append(sim.get_field_point(mp.Hz, res_pt))

    print("Running coupled resonator simulation...")
    sim.run(
        mp.at_every(0.5, record_field),
        until_after_sources=1500,
    )

    # Transmission
    freqs = np.array(mp.get_flux_freqs(trans_mon))
    flux = np.array(mp.get_fluxes(trans_mon))
    freq_ghz_flux = freqs * 300

    # FFT in resonator
    hz_arr = np.array(hz_data, dtype=complex)
    dt = 0.5
    N = len(hz_arr)
    window = np.hanning(N)
    hz_windowed = np.real(hz_arr) * window
    spectrum = np.abs(np.fft.rfft(hz_windowed))
    freqs_meep = np.fft.rfftfreq(N, d=dt)
    freqs_ghz = freqs_meep * 300

    # Field pattern
    hz_field = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Hz)

    sim.reset_meep()
    return {
        "freqs_ghz": freqs_ghz,
        "spectrum": spectrum,
        "freq_ghz_flux": freq_ghz_flux,
        "flux": flux,
        "hz_field": hz_field,
        "cell": (sx, sy),
        "f_qw": f_qw,
        "params": {
            "channel_w": channel_w, "res_len": res_len, "feed_len": feed_len,
            "wall_t": wall_t, "y_feed": y_feed, "y_res_start": y_res_start,
            "y_res_end": y_res_end, "coupling_w": coupling_w,
        },
    }


def plot_results(cavity_res, coupled_res):
    """Generate all plots."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    # === Plot 1: Cavity eigenmode spectrum ===
    fig, ax = plt.subplots(figsize=(10, 6))
    r = cavity_res
    spectrum_db = 20 * np.log10(r["spectrum"] / (r["spectrum"].max() + 1e-30) + 1e-30)
    ax.plot(r["freqs_ghz"], spectrum_db, color="#2c3e50", linewidth=2, label="FFT Spectrum")

    for mode in r["valid_modes"]:
        f_ghz = mode.freq * 300
        Q = mode.Q
        ax.axvline(f_ghz, color="#e74c3c", linestyle="--", alpha=0.7, linewidth=1.5)
        ax.annotate(
            f"f = {f_ghz:.1f} GHz\nQ = {Q:.0f}",
            xy=(f_ghz, -2), xytext=(f_ghz + 1.5, -10),
            fontsize=10, color="#e74c3c",
            arrowprops=dict(arrowstyle="->", color="#e74c3c"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"),
        )

    # Mark analytical modes
    for n in range(1, 5):
        f_n = n * r["f1_exp"] * 300
        if f_n < 45:
            ax.axvline(f_n, color="#27ae60", linestyle=":", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Frequency (GHz)", fontsize=12)
    ax.set_ylabel("Spectrum (dB, normalized)", fontsize=12)
    ax.set_title(
        "Closed CPW Cavity — Eigenmode Spectrum (Harminv Analysis)\n"
        f"L = {r['cav_l']*1000:.0f} μm, W = {r['cav_w']*1000:.0f} μm, "
        f"ε_eff = {r['eps_eff']}  |  Green: analytical modes",
        fontsize=12,
    )
    ax.set_xlim(0, 40)
    ax.set_ylim(-60, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    path = os.path.join(PLOT_DIR, "resonator_eigenmode_spectrum.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # === Plot 2: Cavity field pattern ===
    fig, ax = plt.subplots(figsize=(6, 10))
    sx, sy = r["cell"]
    extent = [-sx/2*1000, sx/2*1000, -sy/2*1000, sy/2*1000]
    vmax = np.max(np.abs(r["hz_field"])) * 0.3
    if vmax < 1e-20:
        vmax = 1.0
    im = ax.imshow(r["hz_field"].T, origin="lower", extent=extent,
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, label="Hz field amplitude", shrink=0.5)

    # Draw cavity walls
    cw, cl, wt = r["cav_w"], r["cav_l"], r["wall_t"]
    for rect_args in [
        (-cw/2 - wt, -cl/2 - wt, wt, cl + 2*wt),  # left
        (cw/2, -cl/2 - wt, wt, cl + 2*wt),          # right
        (-cw/2 - wt, -cl/2 - wt, cw + 2*wt, wt),    # bottom
        (-cw/2 - wt, cl/2, cw + 2*wt, wt),           # top
    ]:
        ax.add_patch(patches.Rectangle(
            (rect_args[0]*1000, rect_args[1]*1000),
            rect_args[2]*1000, rect_args[3]*1000,
            linewidth=1.2, edgecolor="black", facecolor="gray", alpha=0.4,
        ))

    ax.set_xlabel("x (μm)", fontsize=12)
    ax.set_ylabel("y (μm)", fontsize=12)
    ax.set_title("Hz Field — Closed Cavity Eigenmode", fontsize=13)

    path = os.path.join(PLOT_DIR, "resonator_field_pattern.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # === Plot 3: Coupled resonator spectrum + transmission ===
    r2 = coupled_res
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Resonator FFT
    spec_db = 20 * np.log10(r2["spectrum"] / (r2["spectrum"].max() + 1e-30) + 1e-30)
    ax1.plot(r2["freqs_ghz"], spec_db, color="#2c3e50", linewidth=2)
    ax1.axvline(r2["f_qw"]*300, color="#27ae60", linestyle=":", linewidth=2, alpha=0.6)
    ax1.text(r2["f_qw"]*300 + 0.5, -55, f"Analytical f₀ ≈ {r2['f_qw']*300:.1f} GHz",
             fontsize=9, color="#27ae60")
    ax1.set_ylabel("Resonator Spectrum (dB)", fontsize=12)
    ax1.set_title(
        "Coupled Quarter-Wave Resonator — Feedline Transmission\n"
        f"L_res = {r2['params']['res_len']*1000:.0f} μm, "
        f"coupling aperture = {r2['params']['coupling_w']*1000:.0f} μm",
        fontsize=12,
    )
    ax1.set_ylim(-60, 5)
    ax1.grid(True, alpha=0.3)

    # Feedline transmission
    flux_safe = np.where(r2["flux"] > 0, r2["flux"], 1e-30)
    flux_db = 10 * np.log10(flux_safe / flux_safe.max())
    ax2.plot(r2["freq_ghz_flux"], flux_db, color="#e74c3c", linewidth=2)
    ax2.axvline(r2["f_qw"]*300, color="#27ae60", linestyle=":", linewidth=2, alpha=0.6)
    ax2.set_xlabel("Frequency (GHz)", fontsize=12)
    ax2.set_ylabel("Feedline Transmission (dB)", fontsize=12)
    ax2.set_xlim(0, 30)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "resonator_coupled_spectrum.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    # === Plot 4: Coupled resonator field pattern ===
    fig, ax = plt.subplots(figsize=(8, 10))
    sx2, sy2 = r2["cell"]
    extent2 = [-sx2/2*1000, sx2/2*1000, -sy2/2*1000, sy2/2*1000]
    vmax2 = np.max(np.abs(r2["hz_field"])) * 0.3
    if vmax2 < 1e-20:
        vmax2 = 1.0
    im = ax.imshow(r2["hz_field"].T, origin="lower", extent=extent2,
                   cmap="RdBu_r", vmin=-vmax2, vmax=vmax2, aspect="equal")
    plt.colorbar(im, ax=ax, label="Hz field amplitude", shrink=0.6)

    p = r2["params"]
    # Draw structures
    fw = p["feed_len"] + 0.4
    for y, h in [(p["y_feed"] - p["channel_w"]/2 - p["wall_t"], p["wall_t"])]:
        ax.add_patch(patches.Rectangle(
            (-fw/2*1000, y*1000), fw*1000, h*1000,
            linewidth=1, edgecolor="black", facecolor="gray", alpha=0.4))
    # Top wall pieces
    left_w = p["feed_len"]/2 - p["coupling_w"]/2
    for x_start in [-fw/2, p["coupling_w"]/2]:
        ax.add_patch(patches.Rectangle(
            (x_start*1000, (p["y_feed"] + p["channel_w"]/2)*1000),
            left_w*1000, p["wall_t"]*1000,
            linewidth=1, edgecolor="black", facecolor="gray", alpha=0.4))
    # Resonator walls
    for dx in [-p["channel_w"]/2 - p["wall_t"], p["channel_w"]/2]:
        ax.add_patch(patches.Rectangle(
            (dx*1000, p["y_res_start"]*1000),
            p["wall_t"]*1000, p["res_len"]*1000,
            linewidth=1, edgecolor="black", facecolor="gray", alpha=0.4))
    # Top cap
    ax.add_patch(patches.Rectangle(
        ((-p["channel_w"]/2 - p["wall_t"])*1000, p["y_res_end"]*1000),
        (p["channel_w"] + 2*p["wall_t"])*1000, p["wall_t"]*1000,
        linewidth=1, edgecolor="black", facecolor="gray", alpha=0.4))

    ax.set_xlabel("x (μm)", fontsize=12)
    ax.set_ylabel("y (μm)", fontsize=12)
    ax.set_title(
        "Hz Field — Coupled Quarter-Wave Resonator\n"
        "Shorted end (top), coupled to feedline (bottom)",
        fontsize=13,
    )

    path = os.path.join(PLOT_DIR, "resonator_coupled_field.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def run():
    cavity_res = run_closed_cavity()
    coupled_res = run_coupled_resonator()
    plot_results(cavity_res, coupled_res)


if __name__ == "__main__":
    run()
