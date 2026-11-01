"""Layer definitions and CPW cross-section factory for SI test chip."""

import gdsfactory as gf
from gdsfactory.cross_section import Section
from functools import partial

# Layer definitions (GDSII layer, datatype)
LAYER_M1 = (41, 0)       # Metal 1 — CPW conductors and ground
LAYER_VIA = (42, 0)      # Via layer (if needed)
LAYER_M2 = (43, 0)       # Metal 2
LAYER_PAD = (44, 0)      # Probe pad layer
LAYER_LABEL = (100, 0)   # Text labels
LAYER_FRAME = (99, 0)    # Die frame / boundary


def cpw_cross_section(
    width: float = 10.0,
    gap: float = 6.0,
    ground_width: float = 50.0,
) -> gf.CrossSection:
    """Create a CPW cross-section with center conductor and two ground planes.

    Args:
        width: Center conductor width (um).
        gap: Gap between conductor and ground (um).
        ground_width: Width of each ground plane (um).
    """
    # Ground plane offset: center of ground strip is at ±(width/2 + gap + ground_width/2)
    offset_ground = width / 2 + gap + ground_width / 2

    sections = (
        # Center conductor
        Section(
            width=width,
            offset=0,
            layer=LAYER_M1,
            name="_default",
            port_names=("o1", "o2"),
            port_types=("electrical", "electrical"),
        ),
        # Left ground
        Section(
            width=ground_width,
            offset=-offset_ground,
            layer=LAYER_M1,
            name="ground_left",
            port_names=(None, None),
        ),
        # Right ground
        Section(
            width=ground_width,
            offset=+offset_ground,
            layer=LAYER_M1,
            name="ground_right",
            port_names=(None, None),
        ),
    )

    return gf.CrossSection(sections=sections)


# Pre-built cross-sections for standard geometries
cpw_50ohm = partial(cpw_cross_section, width=10.0, gap=6.0)
cpw_high_z = partial(cpw_cross_section, width=5.0, gap=3.0)
cpw_low_z = partial(cpw_cross_section, width=20.0, gap=12.0)
