"""Quarter-wave CPW resonator — the centerpiece structure."""

import gdsfactory as gf
from layers import cpw_cross_section, LAYER_M1, LAYER_LABEL
from cpw_cells import gsg_pad


@gf.cell
def cpw_quarter_wave_resonator(
    feedline_length: float = 800.0,
    resonator_length: float = 500.0,
    coupling_gap: float = 5.0,
    width: float = 10.0,
    gap: float = 6.0,
    ground_width: float = 50.0,
) -> gf.Component:
    """Quarter-wave CPW resonator side-coupled to a feedline.

    A CPW stub resonator capacitively coupled to a through feedline.
    The stub is open at one end and coupled at the other, forming a
    quarter-wave resonator whose fundamental frequency is f = c/(4*L*sqrt(eps_eff)).

    Args:
        feedline_length: Length of the through feedline (um).
        resonator_length: Length of the resonator stub (um).
        coupling_gap: Gap between feedline and resonator (um).
        width: Conductor width (um).
        gap: Conductor-ground gap (um).
        ground_width: Ground plane width per side (um).
    """
    c = gf.Component()

    xs = cpw_cross_section(width=width, gap=gap, ground_width=ground_width)

    # Feedline (horizontal)
    feed_path = gf.path.straight(length=feedline_length)
    feedline = c.add_ref(gf.path.extrude(feed_path, cross_section=xs))

    # Resonator stub (vertical, branching upward from feedline midpoint)
    # Position: coupled to the feedline at the midpoint via a small gap
    res_xs = cpw_cross_section(width=width, gap=gap, ground_width=ground_width)
    res_path = gf.path.straight(length=resonator_length)
    resonator = c.add_ref(gf.path.extrude(res_path, cross_section=res_xs))
    resonator.rotate(90)

    # Place resonator above feedline center, offset by coupling gap
    res_x = feedline_length / 2
    res_y = width / 2 + coupling_gap + width / 2
    resonator.move((res_x, res_y))

    # GSG pads on feedline ends
    pad_pitch = gap + width / 2 + ground_width / 2
    pad_in = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_in.rotate(-90)
    pad_out = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_out.rotate(90)
    pad_out.move((feedline_length, 0))

    # Ports
    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(
        name="o2",
        center=(feedline_length, 0),
        width=width,
        orientation=0,
        layer=LAYER_M1,
    )

    c.add_label(
        text="Lambda/4 Resonator",
        position=(feedline_length / 2, resonator_length + res_y + 30),
        layer=LAYER_LABEL,
    )
    return c
