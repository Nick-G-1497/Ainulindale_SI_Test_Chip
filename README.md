# SI Test Chip — Signal Integrity Characterization Structures

An exploratory learning project to develop a deep understanding of signal integrity concepts — from CPW transmission line design and S-parameter extraction to eigenmode analysis and cryogenic skin effect physics. Built end-to-end with open-source tools: gdsfactory for GDS layout and MEEP for FDTD electromagnetic simulation.

| Test Structure | SI Concept Explored |
|---|---|
| CPW transmission lines (3 geometries) | Impedance characterization, conformal mapping |
| Coupled CPW lines (3 separations) | NEXT/FEXT crosstalk mechanisms |
| Quarter-wave CPW resonator | Eigenmode analysis, Q-factor extraction |
| Impedance discontinuity step | S-parameter extraction (S11), reflection |
| TRL calibration standards | Measurement methodology design |
| Skin depth analysis (300K vs 4K) | Classical vs anomalous skin effect at cryo temps |

## Project Structure

```
si_testchip/
├── 01_layout/                      # GDS layout generation (gdsfactory)
│   ├── layers.py                   # Layer definitions + CPW cross-section factory
│   ├── cpw_cells.py                # CPW lines + GSG probe pads
│   ├── trl_cells.py                # Thru-Reflect-Line calibration standards
│   ├── coupled_cpw_cells.py        # Coupled CPW for crosstalk
│   ├── resonator_cells.py          # Quarter-wave CPW resonator
│   ├── discontinuity_cells.py      # Impedance step structure
│   ├── test_chip.py                # Top-level 5mm × 5mm die assembly
│   └── generate_layout.py          # Main script: generates GDS + PNGs
├── 02_simulation/                  # Electromagnetic simulations (MEEP + analytical)
│   ├── sim_cpw_impedance.py        # Analytical Z0 via Wen 1969 conformal mapping
│   ├── sim_coupled_crosstalk.py    # 2D FDTD NEXT/FEXT vs separation
│   ├── sim_resonator_eigenmode.py  # 2D FDTD eigenmode analysis + Harminv (centerpiece)
│   ├── sim_discontinuity.py        # 2D FDTD S11 extraction with reference normalization
│   ├── sim_skin_depth.py           # Skin depth: 300K vs 4K for cryo-CMOS
│   └── run_all_sims.py             # Runs all simulations sequentially
├── 03_results/plots/               # Generated plots
├── output/                         # Generated GDS and chip PNGs
├── .gitignore
└── README.md
```

## Key Results

### CPW Impedance (Analytical)
- Conformal mapping formula (Wen 1969): `Z0 = 30π/√ε_eff × K(k')/K(k)`
- Verified 50Ω design point at W=10μm, G=6μm on silicon (ε_r=11.9)

### Resonator Eigenmode (MEEP 2D FDTD)
- Closed PEC cavity eigenmode extraction using Harminv
- Detected resonant mode at ~26 GHz with Q > 10⁹ (PEC = lossless)
- Quarter-wave coupled resonator showing resonance in FFT spectrum

### Crosstalk (MEEP 2D FDTD)
- NEXT and FEXT measured for S = 10/20/50 μm separation
- Clear trend: tighter spacing → stronger coupling

### Impedance Discontinuity S11 (MEEP 2D FDTD)
- Two-step normalization: reference run → DUT run with `load_minus_flux_data`
- Analytical: Z1=50Ω → Z2=41Ω gives Γ=0.099 (S11 = -20.1 dB)
- MEEP result compared against analytical prediction

### Skin Depth (Analytical)
- Cu at 10 GHz: δ₃₀₀ₖ = 0.65 μm vs δ₄ₖ = 0.06 μm (10.6× reduction)
- Key insight: at cryogenic temperatures, surface roughness and film quality dominate losses

## Setup

```bash
# Create conda environment
conda create -n si_testchip python=3.11 -y
conda activate si_testchip

# Install MEEP (MUST use conda, not pip)
conda install -c conda-forge pymeep=1.29 -y

# Install layout and analysis tools
pip install gdsfactory[full] gplugins[meep] matplotlib numpy scipy klayout

# Pin numpy for MEEP compatibility
pip install "numpy<2.0"
```

## Usage

```bash
# Generate GDS layout and structure PNGs
python 01_layout/generate_layout.py

# Run all simulations
python 02_simulation/run_all_sims.py

# Or run individual simulations
python 02_simulation/sim_cpw_impedance.py
python 02_simulation/sim_resonator_eigenmode.py
python 02_simulation/sim_coupled_crosstalk.py
python 02_simulation/sim_discontinuity.py
python 02_simulation/sim_skin_depth.py
```

## Tools

- **[gdsfactory](https://gdsfactory.github.io/gdsfactory/)** — GDS layout generation
- **[MEEP](https://meep.readthedocs.io/)** — Open-source FDTD electromagnetic simulation
- **Python 3.11** + NumPy, SciPy, Matplotlib

## Technical Notes

- MEEP units: length in mm, c=1 → frequency in MEEP × 300 = GHz
- PEC metal model — appropriate for superconducting CPW at GHz frequencies
- 2D simulations for laptop-friendly runtimes (minutes, not hours)
- Hz polarization for parallel-plate waveguide models (TEM-like mode)
- Harminv for eigenmode extraction requires long ringdown (~3000+ time units)

## Author

Ada — 2026
