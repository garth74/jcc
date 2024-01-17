# fmt: off
import math

from jccc._numba import njit

_25_POW_7: float =         float(6_103_515_625)
_2PI: float =              float(2.0 * math.pi)
_1_DIV_25: float =         float(1.0 / 25.0)
_30_RAD: float =           math.radians(30)
_6_RAD: float =            math.radians(6)
_63_RAD: float =           math.radians(63)
_KL: float =               float(1.0)
_KC: float =               float(1.0)
_KH: float =               float(1.0)

# fmt: on


@njit(inline="always")
def delta_e_cie2000(
    L1: float,
    a1: float,
    b1: float,
    L2: float,
    a2: float,
    b2: float,
    kL: float = _KL,
    kC: float = _KC,
    kH: float = _KH,
) -> float:
    """
    Calculate ΔE between two CIE LAB values.

    see https://www.hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf
    """
    # distort `a` based on average chroma
    # then convert to lch coordinates from distorted `a`
    # all subsequence calculations are in the new coordinates
    # (often denoted "prime" in the literature)
    Cbar = 0.5 * (math.hypot(a1, b1) + math.hypot(a2, b2))
    c7 = Cbar**7
    G = 0.5 * (1 - math.sqrt(c7 / (c7 + _25_POW_7)))
    scale = 1 + G

    a1scale = a1 * scale
    C1 = math.hypot(a1scale, b1)
    h1: float = math.atan2(b1, a1scale)
    if h1 < 0.0:
        h1 += _2PI

    a2scale = a2 * scale
    C2: float = math.hypot(a2scale, b2)
    h2: float = math.atan2(b2, a2scale)
    if h2 < 0.0:
        h2 += _2PI

    # recall that c, h are polar coordinates.  c==r, h==theta

    # cide2000 has four terms to delta_e:
    # 1) Luminance term
    # 2) Hue term
    # 3) Chroma term
    # 4) hue Rotation term

    # lightness term
    Lbar = 0.5 * (L1 + L2)
    tmp = (Lbar - 50) ** 2
    SL = 1 + 0.015 * tmp / math.sqrt(20 + tmp)
    L_term = (L2 - L1) / (kL * SL)

    # chroma term
    Cbar = 0.5 * (C1 + C2)  # new coordinates
    SC = 1 + 0.045 * Cbar
    C_term = (C2 - C1) / (kC * SC)

    # hue term
    h_diff = h2 - h1
    h_sum = h1 + h2
    CC = C1 * C2

    dH = h_diff
    if CC == 0.0:
        dH = 0.0  # if r == 0, dtheta == 0
    else:
        if h_diff > math.pi:
            dH = h_diff - _2PI
        elif h_diff < -math.pi:
            dH = h_diff + _2PI
        else:
            dH = h_diff

    dH_term = 2.0 * math.sqrt(CC) * math.sin(dH / 2.0)

    if CC == 0.0:
        Hbar = h_sum
    else:
        if math.fabs(h1 - h2) <= math.pi:
            Hbar = h_sum * 0.5
        else:
            if h_sum >= _2PI:
                Hbar = (h_sum - _2PI) * 0.5
            else:
                Hbar = (h_sum + _2PI) * 0.5

    T = (
        1
        - 0.17 * math.cos(Hbar - _30_RAD)
        + 0.24 * math.cos(2 * Hbar)
        + 0.32 * math.cos(3 * Hbar + _6_RAD)
        - 0.20 * math.cos(4 * Hbar - _63_RAD)
    )
    SH = 1 + 0.015 * Cbar * T

    H_term = dH_term / (kH * SH)

    # hue rotation
    c7 = Cbar**7
    Rc = 2 * math.sqrt(c7 / (c7 + _25_POW_7))
    dtheta = _30_RAD * math.exp(-(((math.degrees(Hbar) - 275.0) * _1_DIV_25) ** 2))
    R_term = -math.sin(2.0 * dtheta) * Rc * C_term * H_term

    # put it all together
    dE2 = L_term**2
    dE2 += C_term**2
    dE2 += H_term**2
    dE2 += R_term
    dE2 = math.sqrt(max(dE2, 0.0))
    return dE2
