"""Coupled CPW cells for crosstalk characterization."""

import gdsfactory as gf
from layers import cpw_cross_section, LAYER_M1, LAYER_LABEL
from cpw_cells import gsg_pad


@gf.cell
def coupled_cpw(
    length: float = 500.0,
    width: float = 10.0,
    gap: float = 6.0,
    separation: float = 20.0,
    ground_width: float = 50.0,
) -> gf.Component:
    """Two parallel CPW lines for crosstalk measurement.

    Args:
        length: Coupling length (um).
        width: Conductor width (um).
        gap: Conductor-ground gap (um).
        separation: Edge-to-edge spacing between the two CPW ground planes (um).
        ground_width: Ground plane width (um).
    """
    c = gf.Component()

    xs = cpw_cross_section(width=width, gap=gap, ground_width=ground_width)

    # Aggressor line (top)
    path = gf.path.straight(length=length)
    y_offset = separation / 2 + ground_width + gap + width / 2
    agg = c.add_ref(gf.path.extrude(path, cross_section=xs))
    agg.move((0, y_offset))

    # Victim line (bottom)
    vic = c.add_ref(gf.path.extrude(path, cross_section=xs))
    vic.move((0, -y_offset))

    # Add probe pads for all 4 ports
    pad_pitch = gap + width / 2 + ground_width / 2
    for label, pos, rot in [
        ("Agg_in", (0, y_offset), -90),
        ("Agg_out", (length, y_offset), 90),
        ("Vic_near", (0, -y_offset), -90),
        ("Vic_far", (length, -y_offset), 90),
    ]:
        pad = c.add_ref(gsg_pad(pitch=pad_pitch))
        pad.rotate(rot)
        pad.move(pos)

    # Ports
    c.add_port(name="agg_in", center=(0, y_offset), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(name="agg_out", center=(length, y_offset), width=width, orientation=0, layer=LAYER_M1)
    c.add_port(name="vic_near", center=(0, -y_offset), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(name="vic_far", center=(length, -y_offset), width=width, orientation=0, layer=LAYER_M1)

    return c


@gf.cell
def coupled_cpw_set() -> gf.Component:
    """Coupled CPW structures at three separations for crosstalk sweep."""
    c = gf.Component()

    separations = [10.0, 20.0, 50.0]
    y_pos = 0

    for sep in separations:
        ref = c.add_ref(coupled_cpw(separation=sep))
        ref.move((0, y_pos))
        c.add_label(
            text=f"Coupled S={sep}um",
            position=(250, y_pos + 200),
            layer=LAYER_LABEL,
        )
        y_pos += 500

    return c
