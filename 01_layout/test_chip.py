"""Top-level die assembly for the SI test chip."""

import gdsfactory as gf
from layers import LAYER_FRAME, LAYER_LABEL
from cpw_cells import cpw_line_set
from trl_cells import trl_cal_set
from coupled_cpw_cells import coupled_cpw_set
from resonator_cells import cpw_quarter_wave_resonator
from discontinuity_cells import cpw_impedance_step


@gf.cell
def si_test_chip(
    die_width: float = 5000.0,
    die_height: float = 5000.0,
) -> gf.Component:
    """5mm x 5mm SI test chip with all structures.

    Manual placement for reliable layout (gf.pack can be fragile).
    """
    c = gf.Component()

    # Die frame
    frame = c.add_ref(
        gf.components.rectangle(size=(die_width, die_height), layer=LAYER_FRAME)
    )
    frame.move((-die_width / 2, -die_height / 2))

    # Place structures with manual offsets
    margin = 200.0

    # Die labels
    c.add_label(
        text="SI TEST CHIP v1.0",
        position=(0, die_height / 2 - 100),
        layer=LAYER_LABEL,
    )
    # Chip name, date, and designer — bottom-right corner
    c.add_label(
        text="AINULINDALE",
        position=(die_width / 2 - margin, -die_height / 2 + 120),
        layer=LAYER_LABEL,
    )
    c.add_label(
        text="2026-11-01",
        position=(die_width / 2 - margin, -die_height / 2 + 80),
        layer=LAYER_LABEL,
    )
    c.add_label(
        text="Ada",
        position=(die_width / 2 - margin, -die_height / 2 + 40),
        layer=LAYER_LABEL,
    )

    # 1. CPW line set (top-left quadrant)
    cpw_set = c.add_ref(cpw_line_set())
    cpw_set.move((-die_width / 2 + margin, die_height / 2 - 900))

    # 2. TRL calibration standards (top-right quadrant)
    trl = c.add_ref(trl_cal_set())
    trl.move((500, die_height / 2 - 900))

    # 3. Quarter-wave resonator (center — the centerpiece)
    resonator = c.add_ref(cpw_quarter_wave_resonator())
    resonator.move((-die_width / 2 + margin, -200))

    # 4. Coupled CPW set (bottom-left)
    coupled = c.add_ref(coupled_cpw_set())
    coupled.move((-die_width / 2 + margin, -die_height / 2 + margin))

    # 5. Impedance discontinuity (bottom-right)
    step = c.add_ref(cpw_impedance_step())
    step.move((500, -die_height / 2 + margin + 500))

    return c
