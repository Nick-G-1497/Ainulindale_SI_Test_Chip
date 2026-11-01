"""CPW impedance discontinuity (step) cell."""

import gdsfactory as gf
from layers import cpw_cross_section, LAYER_M1, LAYER_LABEL
from cpw_cells import gsg_pad


@gf.cell
def cpw_impedance_step(
    length1: float = 300.0,
    length2: float = 300.0,
    width1: float = 10.0,
    gap1: float = 6.0,
    width2: float = 20.0,
    gap2: float = 6.0,
    ground_width: float = 50.0,
) -> gf.Component:
    """Abrupt CPW impedance step (50-ohm to ~35-ohm).

    Two CPW sections of different geometry joined at a step discontinuity.
    Used for measuring S11 reflection from impedance mismatch.

    Args:
        length1: Length of first (50Ω) section (um).
        length2: Length of second (~35Ω) section (um).
        width1/gap1: First section geometry.
        width2/gap2: Second section geometry.
        ground_width: Ground plane width (um).
    """
    c = gf.Component()

    # Section 1 (50 ohm)
    xs1 = cpw_cross_section(width=width1, gap=gap1, ground_width=ground_width)
    path1 = gf.path.straight(length=length1)
    sec1 = c.add_ref(gf.path.extrude(path1, cross_section=xs1))

    # Section 2 (lower impedance)
    xs2 = cpw_cross_section(width=width2, gap=gap2, ground_width=ground_width)
    path2 = gf.path.straight(length=length2)
    sec2 = c.add_ref(gf.path.extrude(path2, cross_section=xs2))
    sec2.move((length1, 0))

    total = length1 + length2

    # GSG pads
    pad_pitch1 = gap1 + width1 / 2 + ground_width / 2
    pad_in = c.add_ref(gsg_pad(pitch=pad_pitch1))
    pad_in.rotate(-90)

    pad_pitch2 = gap2 + width2 / 2 + ground_width / 2
    pad_out = c.add_ref(gsg_pad(pitch=pad_pitch2))
    pad_out.rotate(90)
    pad_out.move((total, 0))

    # Ports
    c.add_port(name="o1", center=(0, 0), width=width1, orientation=180, layer=LAYER_M1)
    c.add_port(
        name="o2", center=(total, 0), width=width2, orientation=0, layer=LAYER_M1
    )

    c.add_label(
        text="Z-step 50->41 ohm",
        position=(total / 2, 60),
        layer=LAYER_LABEL,
    )
    return c
