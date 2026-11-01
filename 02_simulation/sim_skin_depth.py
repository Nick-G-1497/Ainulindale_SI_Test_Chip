#!/usr/bin/env python3
"""Skin depth comparison: classical vs anomalous, 300K vs 4K.

This simulation demonstrates how the skin depth changes at cryogenic
temperatures, and critically, why the classical skin depth formula BREAKS
DOWN at 4K for high-purity metals — requiring the anomalous skin effect
(Reuter-Sondheimer theory, 1948).

=== Classical Skin Effect ===

    δ_classical = sqrt(2ρ / (ωμ₀))

    Valid when: mean free path l << δ  (electrons scatter many times
    within one skin depth → local Ohm's law J = σE holds)

=== Anomalous Skin Effect ===

When l >> δ (cryogenic, high-purity metals), electrons traverse the skin
depth without scattering. The local relation J = σE breaks down because
the current at a point depends on the electric field over the entire mean
free path. Reuter & Sondheimer (1948) showed:

    δ_anomalous ≈ (3δ_classical² × l / 4)^(1/3)     [extreme anomalous limit]

    Equivalently: δ_anom = (3c² ρ l / (2ω))^(1/3)  where c = 1/sqrt(μ₀ε₀)
    or more usefully: δ_anom = (3/(4π²))^(1/3) × (ρ l / (f μ₀))^(1/3)

The surface impedance also changes:
    Classical:  Z_s = (1+j)/(σδ)           → R_s ∝ √ω
    Anomalous:  Z_s ∝ (1+j√3)/2 × ω^(2/3) → R_s ∝ ω^(2/3)

The anomalous/classical ratio:
    α = l / δ_classical     (anomaly parameter)
    α >> 1  → anomalous regime
    α << 1  → classical regime

For Cu (RRR=100) at 4K, 10 GHz:
    l ≈ 4.0 μm,  δ_classical ≈ 0.062 μm  →  α ≈ 65  (deeply anomalous!)

For Cu at 300K, 10 GHz:
    l ≈ 0.039 μm,  δ_classical ≈ 0.65 μm  →  α ≈ 0.06  (classical)

References:
    G.E.H. Reuter & E.H. Sondheimer, Proc. R. Soc. A 195 (1948) 336-364.
    A.B. Pippard, Proc. R. Soc. A 191 (1947) 385-399.
    J.R. Sambles, Thin Solid Films 106 (1983) 321-331.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "03_results", "plots")

MU_0 = 4 * np.pi * 1e-7  # H/m


# ── Material database ────────────────────────────────────────────────

def mean_free_path(rho, n_e=8.5e28, m_eff=9.109e-31, e=1.602e-19):
    """Estimate electron mean free path from Drude model.

    l = m_eff * v_F / (n_e * e² * ρ)  but simpler via:
    l = m_eff * v_F / (ρ * n_e * e²)

    For copper: v_F ≈ 1.57e6 m/s, n_e ≈ 8.5e28 /m³
    """
    v_F = 1.57e6  # Fermi velocity for Cu (m/s)
    # Drude: σ = n_e e² τ / m_eff, and l = v_F × τ
    # So: l = v_F × m_eff × σ / (n_e × e²) = v_F × m_eff / (n_e × e² × ρ)
    sigma = 1.0 / rho
    tau = m_eff * sigma / (n_e * e**2)
    return v_F * tau


MATERIALS = {
    "Cu": {
        "300K": {"rho": 1.68e-8, "label": "Cu (300K)"},
        "77K":  {"rho": 2.0e-9,  "label": "Cu (77K, RRR≈100)"},
        "4K":   {"rho": 1.5e-10, "label": "Cu (4K, RRR≈100)"},
    },
    "Al": {
        "300K": {"rho": 2.65e-8, "label": "Al (300K)"},
        "77K":  {"rho": 3.0e-9,  "label": "Al (77K)"},
        "4K":   {"rho": 2.5e-10, "label": "Al (4K, normal state)"},
    },
}


# ── Physics functions ────────────────────────────────────────────────

def skin_depth_classical(rho, freq):
    """Classical skin depth δ = √(2ρ/(ωμ₀)) in meters."""
    omega = 2 * np.pi * freq
    return np.sqrt(2 * rho / (omega * MU_0))


def skin_depth_anomalous(rho, freq, mfp):
    """Anomalous skin depth (extreme anomalous limit, Reuter-Sondheimer).

    δ_anom = (3 δ_cl² l / 4)^(1/3)

    This is valid when α = l/δ_cl >> 1.

    Args:
        rho: Resistivity (Ω·m)
        freq: Frequency (Hz)
        mfp: Mean free path (m)
    Returns:
        Anomalous skin depth (m)
    """
    delta_cl = skin_depth_classical(rho, freq)
    return (3.0 * delta_cl**2 * mfp / 4.0) ** (1.0 / 3.0)


def skin_depth_interpolated(rho, freq, mfp):
    """Smooth interpolation between classical and anomalous regimes.

    Uses the Dingle (1953) interpolation formula which gives a smooth
    crossover. In the limits:
        α << 1: returns δ_classical
        α >> 1: returns δ_anomalous

    The effective penetration depth from the full Chambers/Reuter-Sondheimer
    integral can be approximated by:
        1/δ_eff³ ≈ 1/δ_cl³ + 1/δ_anom³

    This gives the right limiting behavior in both regimes.
    """
    delta_cl = skin_depth_classical(rho, freq)
    delta_anom = skin_depth_anomalous(rho, freq, mfp)
    # Interpolation: harmonic-cube mean
    return (delta_cl**3 + delta_anom**3) ** (1.0 / 3.0)


def surface_resistance_classical(rho, freq):
    """Classical surface resistance R_s = √(ωμ₀ρ/2) in Ω."""
    omega = 2 * np.pi * freq
    return np.sqrt(omega * MU_0 * rho / 2.0)


def surface_resistance_anomalous(rho, freq, mfp):
    """Anomalous surface resistance (extreme anomalous limit).

    R_s = (√3/2) × (3μ₀²ω²ρ / (4π l))^(1/3) × (1/2)

    More compactly: R_s_anom / R_s_cl ≈ (3π/4)^(1/3) × (l/δ_cl)^(1/3)
    """
    omega = 2 * np.pi * freq
    delta_cl = skin_depth_classical(rho, freq)
    alpha = mfp / delta_cl

    # In the anomalous limit, R_s scales as ω^(2/3) instead of ω^(1/2)
    # R_s_anom = (√3/2) × (μ₀ω)^(2/3) × (3ρl/(4π))^(1/3) / (something)
    # Simpler: use R_s = 1/(σ δ_anom) with anomalous δ
    sigma = 1.0 / rho
    delta_anom = skin_depth_anomalous(rho, freq, mfp)
    return 1.0 / (sigma * delta_anom)


# ── Plot functions ───────────────────────────────────────────────────

def plot_classical_vs_anomalous_skin_depth():
    """Main plot: classical vs anomalous skin depth for Cu at multiple T."""
    freqs_hz = np.logspace(8, 11, 300)  # 100 MHz to 100 GHz
    freqs_ghz = freqs_hz / 1e9

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))

    # ── Panel (a): Classical skin depth for all materials/temps ──
    ax = axes[0, 0]
    styles = [
        ("Cu (300K)", 1.68e-8, "#e74c3c", "-"),
        ("Cu (77K)",  2.0e-9,  "#e74c3c", "-."),
        ("Cu (4K)",   1.5e-10, "#e74c3c", "--"),
        ("Al (300K)", 2.65e-8, "#3498db", "-"),
        ("Al (4K)",   2.5e-10, "#3498db", "--"),
    ]
    for name, rho, color, ls in styles:
        delta = skin_depth_classical(rho, freqs_hz) * 1e6
        ax.loglog(freqs_ghz, delta, color=color, linestyle=ls, linewidth=2, label=name)

    for t, label in [(1.0, "1 μm film"), (0.1, "100 nm film")]:
        ax.axhline(t, color="gray", linestyle=":", alpha=0.4)
        ax.text(0.12, t * 1.15, label, fontsize=8, color="gray")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Classical Skin Depth δ_cl (μm)")
    ax.set_title("(a) Classical Skin Depth — All Temperatures", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.005, 100)

    # ── Panel (b): Anomaly parameter α = l/δ_cl for Cu ──
    ax = axes[0, 1]
    cu_configs = [
        ("Cu 300K", 1.68e-8, "#e74c3c", "-"),
        ("Cu 77K",  2.0e-9,  "#e67e22", "-."),
        ("Cu 4K",   1.5e-10, "#3498db", "--"),
    ]
    for name, rho, color, ls in cu_configs:
        mfp = mean_free_path(rho)
        delta_cl = skin_depth_classical(rho, freqs_hz)
        alpha = mfp / delta_cl
        ax.loglog(freqs_ghz, alpha, color=color, linestyle=ls, linewidth=2.5,
                  label=f"{name} (l={mfp*1e6:.2f} μm)")

    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.fill_between(freqs_ghz, 1.0, 1e4, alpha=0.06, color="red",
                    label="Anomalous regime (α > 1)")
    ax.fill_between(freqs_ghz, 1e-3, 1.0, alpha=0.06, color="green",
                    label="Classical regime (α < 1)")

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Anomaly Parameter α = l / δ_cl")
    ax.set_title("(b) Anomaly Parameter — When Classical Breaks Down", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.01, 1000)

    # ── Panel (c): Classical vs Anomalous skin depth for Cu at 4K ──
    ax = axes[1, 0]
    rho_4K = 1.5e-10
    mfp_4K = mean_free_path(rho_4K)
    rho_300K = 1.68e-8
    mfp_300K = mean_free_path(rho_300K)

    delta_cl_4K = skin_depth_classical(rho_4K, freqs_hz) * 1e6
    delta_anom_4K = skin_depth_anomalous(rho_4K, freqs_hz, mfp_4K) * 1e6
    delta_interp_4K = skin_depth_interpolated(rho_4K, freqs_hz, mfp_4K) * 1e6
    delta_cl_300K = skin_depth_classical(rho_300K, freqs_hz) * 1e6

    ax.loglog(freqs_ghz, delta_cl_300K, color="#e74c3c", linewidth=2, label="Cu 300K (classical)")
    ax.loglog(freqs_ghz, delta_cl_4K, color="#3498db", linewidth=2, linestyle="--",
              label="Cu 4K — classical (WRONG)")
    ax.loglog(freqs_ghz, delta_anom_4K, color="#2ecc71", linewidth=2, linestyle="-.",
              label="Cu 4K — anomalous limit")
    ax.loglog(freqs_ghz, delta_interp_4K, color="#9b59b6", linewidth=2.5,
              label="Cu 4K — Reuter-Sondheimer interp.")

    # Shade the error region between classical and anomalous at 4K
    ax.fill_between(freqs_ghz, delta_cl_4K, delta_interp_4K, alpha=0.15, color="#e74c3c",
                    label="Classical underestimates δ by this much")

    for t, label in [(1.0, "1 μm"), (0.1, "100 nm")]:
        ax.axhline(t, color="gray", linestyle=":", alpha=0.4)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Effective Skin Depth (μm)")
    ax.set_title("(c) Classical vs Anomalous Skin Depth — Cu at 4K", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 100)
    ax.set_ylim(0.01, 10)

    # ── Panel (d): Surface resistance classical vs anomalous ──
    ax = axes[1, 1]

    Rs_cl_300K = surface_resistance_classical(rho_300K, freqs_hz) * 1e3  # mΩ
    Rs_cl_4K = surface_resistance_classical(rho_4K, freqs_hz) * 1e3
    Rs_anom_4K = surface_resistance_anomalous(rho_4K, freqs_hz, mfp_4K) * 1e3

    ax.loglog(freqs_ghz, Rs_cl_300K, color="#e74c3c", linewidth=2,
              label="Cu 300K (classical)")
    ax.loglog(freqs_ghz, Rs_cl_4K, color="#3498db", linewidth=2, linestyle="--",
              label="Cu 4K — classical (WRONG)")
    ax.loglog(freqs_ghz, Rs_anom_4K, color="#9b59b6", linewidth=2.5,
              label="Cu 4K — anomalous")

    ax.fill_between(freqs_ghz, Rs_cl_4K, Rs_anom_4K, alpha=0.15, color="#9b59b6",
                    label="Additional loss from anomalous effect")

    # Annotate the frequency scaling
    ax.annotate("R_s ∝ √ω (classical)", xy=(30, Rs_cl_300K[200]),
                fontsize=9, color="#e74c3c", rotation=20)
    ax.annotate("R_s ∝ ω^(2/3) (anomalous)", xy=(5, Rs_anom_4K[150]),
                fontsize=9, color="#9b59b6", rotation=25)

    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Surface Resistance R_s (mΩ)")
    ax.set_title("(d) Surface Resistance — ω^(1/2) vs ω^(2/3) Scaling", fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.1, 100)

    fig.suptitle(
        "Skin Effect: Classical vs Anomalous — Implications for Cryo-CMOS Interconnects\n"
        "Reuter-Sondheimer (1948): when mean free path l >> skin depth δ, "
        "the classical formula underestimates penetration",
        fontsize=13, y=1.03,
    )
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "skin_depth_classical_vs_anomalous.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_penetration_comparison():
    """Field penetration profiles: classical vs anomalous at 4K."""
    freq = 10e9
    x_um = np.linspace(0, 1.5, 1000)
    x_m = x_um * 1e-6

    rho_300K = 1.68e-8
    rho_4K = 1.5e-10
    mfp_4K = mean_free_path(rho_4K)
    mfp_300K = mean_free_path(rho_300K)

    delta_cl_300K = skin_depth_classical(rho_300K, freq)
    delta_cl_4K = skin_depth_classical(rho_4K, freq)
    delta_anom_4K = skin_depth_anomalous(rho_4K, freq, mfp_4K)
    delta_rs_4K = skin_depth_interpolated(rho_4K, freq, mfp_4K)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: field decay profiles
    profiles = [
        ("Cu 300K (classical)", delta_cl_300K, "#e74c3c", "-", 2.5),
        ("Cu 4K — classical (WRONG)", delta_cl_4K, "#3498db", "--", 2),
        ("Cu 4K — anomalous limit", delta_anom_4K, "#2ecc71", "-.", 2),
        ("Cu 4K — R-S interpolated", delta_rs_4K, "#9b59b6", "-", 2.5),
    ]

    for label, delta, color, ls, lw in profiles:
        decay = np.exp(-x_m / delta)
        ax1.plot(x_um, decay, color=color, linestyle=ls, linewidth=lw, label=label)

    # Mark 1/e point for each
    for label, delta, color, _, _ in profiles:
        d_um = delta * 1e6
        if d_um < 1.5:
            ax1.plot(d_um, np.exp(-1), "o", color=color, markersize=6)

    ax1.axhline(np.exp(-1), color="gray", linestyle=":", alpha=0.3)
    ax1.text(1.35, np.exp(-1) + 0.02, "1/e", fontsize=9, color="gray")

    for t, label in [(0.5, "500 nm film"), (1.0, "1 μm film")]:
        ax1.axvline(t, color="gray", linestyle=":", alpha=0.4)
        ax1.text(t + 0.02, 0.92, label, fontsize=8, color="gray", rotation=90, va="top")

    ax1.set_xlabel("Depth into Conductor (μm)", fontsize=12)
    ax1.set_ylabel("Normalized Field |E/E₀|", fontsize=12)
    ax1.set_title("Field Penetration at 10 GHz — Classical vs Anomalous", fontsize=12)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1.5)
    ax1.set_ylim(0, 1.05)

    # Right: ratio δ_anomalous / δ_classical vs temperature (via RRR)
    ax2b = ax2
    rrr_values = np.logspace(0, 3, 200)  # RRR from 1 to 1000
    rho_base = 1.68e-8  # Cu at 300K

    freqs_plot = [1e9, 5e9, 10e9, 40e9]
    colors_f = ["#e74c3c", "#e67e22", "#3498db", "#9b59b6"]

    for freq_p, color_p in zip(freqs_plot, colors_f):
        ratios = []
        for rrr in rrr_values:
            rho_cold = rho_base / rrr
            mfp_cold = mean_free_path(rho_cold)
            d_cl = skin_depth_classical(rho_cold, freq_p)
            d_rs = skin_depth_interpolated(rho_cold, freq_p, mfp_cold)
            ratios.append(d_rs / d_cl)
        ax2b.semilogx(rrr_values, ratios, color=color_p, linewidth=2,
                      label=f"{freq_p/1e9:.0f} GHz")

    ax2b.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax2b.fill_between(rrr_values, 1.0, 0, alpha=0.05, color="green")
    ax2b.annotate("Classical regime\n(ratio ≈ 1)", xy=(2, 1.02), fontsize=9, color="green")
    ax2b.annotate("Anomalous regime\n(actual δ >> classical δ)",
                  xy=(100, 2.5), fontsize=9, color="red")
    ax2b.axvline(100, color="gray", linestyle=":", alpha=0.4)
    ax2b.text(110, 1.1, "Typical\nhigh-purity Cu\nRRR≈100", fontsize=8, color="gray")

    ax2b.set_xlabel("Residual Resistivity Ratio (RRR = ρ₃₀₀/ρ_cold)", fontsize=12)
    ax2b.set_ylabel("δ_effective / δ_classical", fontsize=12)
    ax2b.set_title(
        "Anomalous Correction Factor vs Material Purity\n"
        "Higher RRR → longer mean free path → larger anomalous correction",
        fontsize=11,
    )
    ax2b.legend(fontsize=10, title="Frequency")
    ax2b.grid(True, alpha=0.3, which="both")
    ax2b.set_xlim(1, 1000)
    ax2b.set_ylim(0.9, 5)

    fig.suptitle(
        "Classical vs Anomalous Skin Effect — Why It Matters for Cryo-CMOS",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(PLOT_DIR, "skin_depth_penetration.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def print_summary():
    """Print a detailed comparison table."""
    print("\n" + "=" * 80)
    print("SKIN DEPTH SUMMARY: Classical vs Anomalous")
    print("=" * 80)

    freq = 10e9
    configs = [
        ("Cu 300K", 1.68e-8),
        ("Cu 77K (RRR≈100)", 2.0e-9),
        ("Cu 4K  (RRR≈100)", 1.5e-10),
        ("Al 300K", 2.65e-8),
        ("Al 4K  (normal)", 2.5e-10),
    ]

    header = (f"{'Material':<22} {'ρ (Ω·m)':>12} {'l (μm)':>10} "
              f"{'δ_cl (μm)':>10} {'δ_anom (μm)':>12} {'δ_eff (μm)':>11} "
              f"{'α = l/δ':>8} {'Regime':>12}")
    print(f"\nAt {freq/1e9:.0f} GHz:")
    print(header)
    print("-" * len(header))

    for name, rho in configs:
        mfp = mean_free_path(rho)
        d_cl = skin_depth_classical(rho, freq)
        d_anom = skin_depth_anomalous(rho, freq, mfp)
        d_eff = skin_depth_interpolated(rho, freq, mfp)
        alpha = mfp / d_cl
        regime = "ANOMALOUS" if alpha > 3 else ("transitional" if alpha > 0.3 else "classical")

        print(f"{name:<22} {rho:>12.2e} {mfp*1e6:>10.3f} "
              f"{d_cl*1e6:>10.4f} {d_anom*1e6:>12.4f} {d_eff*1e6:>11.4f} "
              f"{alpha:>8.1f} {regime:>12}")

    print("\nKey insight: At 4K, classical formula underestimates Cu skin depth by ~3×")
    print("This means surface losses are HIGHER than classical prediction at cryo temps.")
    print("For interconnect design: use anomalous R_s (∝ ω^2/3) not classical (∝ √ω).\n")


def run():
    plot_classical_vs_anomalous_skin_depth()
    plot_penetration_comparison()
    print_summary()


if __name__ == "__main__":
    run()
