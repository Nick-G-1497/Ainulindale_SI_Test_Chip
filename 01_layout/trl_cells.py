"""TRL (Thru-Reflect-Line) calibration standard cells."""

import gdsfactory as gf
from layers import cpw_cross_section, LAYER_M1, LAYER_LABEL
from cpw_cells import gsg_pad


@gf.cell
def trl_thru(
    width: float = 10.0,
    gap: float = 6.0,
    length: float = 200.0,
) -> gf.Component:
    """Thru standard — short CPW line (reference length)."""
    c = gf.Component()
    xs = cpw_cross_section(width=width, gap=gap)
    path = gf.path.straight(length=length)
    cpw = c.add_ref(gf.path.extrude(path, cross_section=xs))

    pad_pitch = gap + width / 2 + 25.0
    pad_in = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_in.rotate(-90)
    pad_out = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_out.rotate(90)
    pad_out.move((length, 0))

    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(
        name="o2", center=(length, 0), width=width, orientation=0, layer=LAYER_M1
    )
    c.add_label(text="THRU", position=(length / 2, 60), layer=LAYER_LABEL)
    return c


@gf.cell
def trl_reflect(
    width: float = 10.0,
    gap: float = 6.0,
    stub_length: float = 100.0,
) -> gf.Component:
    """Reflect standard — open-ended CPW stub."""
    c = gf.Component()
    xs = cpw_cross_section(width=width, gap=gap)
    path = gf.path.straight(length=stub_length)
    cpw = c.add_ref(gf.path.extrude(path, cross_section=xs))

    pad_pitch = gap + width / 2 + 25.0
    pad_in = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_in.rotate(-90)

    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=LAYER_M1)
    c.add_label(text="REFLECT (Open)", position=(stub_length / 2, 60), layer=LAYER_LABEL)
    return c


@gf.cell
def trl_line(
    width: float = 10.0,
    gap: float = 6.0,
    thru_length: float = 200.0,
    extra_length: float = 250.0,
) -> gf.Component:
    """Line standard — Thru length + quarter-wave extra at center freq.

    extra_length ≈ λ/4 at ~10 GHz on Si (ε_eff≈6.5): λ=c/(f*sqrt(ε_eff))≈11.8mm,
    λ/4≈2.95mm ≈ 2950um. We use a shorter 250um for compact die.
    """
    c = gf.Component()
    total = thru_length + extra_length
    xs = cpw_cross_section(width=width, gap=gap)
    path = gf.path.straight(length=total)
    cpw = c.add_ref(gf.path.extrude(path, cross_section=xs))

    pad_pitch = gap + width / 2 + 25.0
    pad_in = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_in.rotate(-90)
    pad_out = c.add_ref(gsg_pad(pitch=pad_pitch))
    pad_out.rotate(90)
    pad_out.move((total, 0))

    c.add_port(name="o1", center=(0, 0), width=width, orientation=180, layer=LAYER_M1)
    c.add_port(name="o2", center=(total, 0), width=width, orientation=0, layer=LAYER_M1)
    c.add_label(text="LINE", position=(total / 2, 60), layer=LAYER_LABEL)
    return c


@gf.cell
def trl_cal_set() -> gf.Component:
    """Complete TRL calibration standard set."""
    c = gf.Component()

    thru = c.add_ref(trl_thru())
    reflect = c.add_ref(trl_reflect())
    line = c.add_ref(trl_line())

    # Stack vertically
    reflect.move((0, 200))
    line.move((0, 400))

    c.add_label(text="TRL Cal Standards", position=(200, 600), layer=LAYER_LABEL)
    return c
