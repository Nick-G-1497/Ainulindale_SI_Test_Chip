"""CPW transmission line cells with GSG probe pads."""

import gdsfactory as gf
from gdsfactory.typings import LayerSpec

from layers import cpw_cross_section, LAYER_M1, LAYER_PAD, LAYER_LABEL


@gf.cell
def gsg_pad(
    pitch: float = 100.0,
    pad_width: float = 80.0,
    pad_height: float = 80.0,
    layer_pad: LayerSpec = LAYER_PAD,
) -> gf.Component:
    """Ground-Signal-Ground probe pad triplet.

    Args:
        pitch: Center-to-center pad spacing (um).
        pad_width: Individual pad width (um).
        pad_height: Individual pad height (um).
    """
    c = gf.Component()

    for i, name in enumerate(["G1", "S", "G2"]):
        x_offset = (i - 1) * pitch
        pad = c.add_ref(
            gf.components.rectangle(size=(pad_width, pad_height), layer=layer_pad)
        )
        pad.move((x_offset - pad_width / 2, 0))

    # Add port at center pad for connection
    c.add_port(
        name="signal",
        center=(0, pad_height / 2),
        width=pad_width,
        orientation=90,
        layer=layer_pad,
    )
    return c


@gf.cell
def cpw_line(
    length: float = 500.0,
    width: float = 10.0,
    gap: float = 6.0,
    ground_width: float = 50.0,
    with_pads: bool = True,
) -> gf.Component:
    """CPW transmission line with optional GSG pads at both ends.

    Args:
        length: Line length (um).
        width: Center conductor width (um).
        gap: Conductor-ground gap (um).
        ground_width: Ground plane width (um).
        with_pads: Whether to add GSG probe pads.
    """
    c = gf.Component()

    xs = cpw_cross_section(width=width, gap=gap, ground_width=ground_width)
    path = gf.path.straight(length=length)
    cpw = c.add_ref(gf.path.extrude(path, cross_section=xs))

    if with_pads:
        pad_pitch = gap + width / 2 + ground_width / 2
        # Input pad
        pad_in = c.add_ref(gsg_pad(pitch=pad_pitch))
        pad_in.rotate(-90)
        pad_in.move((0, 0))

        # Output pad
        pad_out = c.add_ref(gsg_pad(pitch=pad_pitch))
        pad_out.rotate(90)
        pad_out.move((length, 0))

    # Add electrical ports
    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(
        name="o2", center=(length, 0), width=width, orientation=0, layer=LAYER_M1
    )
    return c


@gf.cell
def cpw_line_set() -> gf.Component:
    """Set of three CPW lines: high-Z, 50-ohm, and low-Z."""
    c = gf.Component()

    configs = [
        {"width": 5.0, "gap": 3.0, "label": "High-Z (W=5/G=3)"},
        {"width": 10.0, "gap": 6.0, "label": "50ohm (W=10/G=6)"},
        {"width": 20.0, "gap": 12.0, "label": "Low-Z (W=20/G=12)"},
    ]

    y_offset = 0
    spacing = 250.0

    for cfg in configs:
        line = c.add_ref(
            cpw_line(length=500.0, width=cfg["width"], gap=cfg["gap"])
        )
        line.move((0, y_offset))
        # Add label
        c.add_label(
            text=cfg["label"],
            position=(250, y_offset + 60),
            layer=LAYER_LABEL,
        )
        y_offset += spacing

    return c
