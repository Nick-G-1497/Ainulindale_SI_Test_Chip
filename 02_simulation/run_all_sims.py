#!/usr/bin/env python3
"""Run all SI test chip simulations sequentially."""

import os
import sys
import time

# Add simulation dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_all():
    total_start = time.time()

    print("=" * 60)
    print("SI TEST CHIP — Full Simulation Suite")
    print("=" * 60)

    # 1. Analytical CPW impedance
    print("\n\n>>> 1/5: CPW Impedance (Analytical)")
    print("-" * 40)
    t0 = time.time()
    import sim_cpw_impedance
    sim_cpw_impedance.run()
    print(f"  Completed in {time.time()-t0:.1f}s")

    # 2. Coupled CPW crosstalk
    print("\n\n>>> 2/5: Coupled CPW Crosstalk (MEEP 2D)")
    print("-" * 40)
    t0 = time.time()
    import sim_coupled_crosstalk
    sim_coupled_crosstalk.run()
    print(f"  Completed in {time.time()-t0:.1f}s")

    # 3. Quarter-wave resonator eigenmode (CENTERPIECE)
    print("\n\n>>> 3/5: Quarter-Wave Resonator Eigenmode (MEEP 2D)")
    print("-" * 40)
    t0 = time.time()
    import sim_resonator_eigenmode
    sim_resonator_eigenmode.run()
    print(f"  Completed in {time.time()-t0:.1f}s")

    # 4. Impedance discontinuity S11
    print("\n\n>>> 4/5: Impedance Discontinuity S11 (MEEP 2D)")
    print("-" * 40)
    t0 = time.time()
    import sim_discontinuity
    sim_discontinuity.run()
    print(f"  Completed in {time.time()-t0:.1f}s")

    # 5. Skin depth comparison (300K vs 4K)
    print("\n\n>>> 5/5: Skin Depth 300K vs 4K")
    print("-" * 40)
    t0 = time.time()
    import sim_skin_depth
    sim_skin_depth.run()
    print(f"  Completed in {time.time()-t0:.1f}s")

    total = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"All simulations complete! Total time: {total:.1f}s")
    print("=" * 60)
    print("\nPlots saved to 03_results/plots/")


if __name__ == "__main__":
    run_all()
