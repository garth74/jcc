# type: ignore
# ruff: noqa: E741
"""
This contains low level conversion functions.

Adapted from https://github.com/CairX/convert-colors-py
https://www.easyrgb.com/en/math.php
"""

# fmt: off

import numpy as np
from numba.core.errors import NumbaTypeError
from numba.extending import overload

from jccc._numba import njit, prange, register_jitable, types

_1_DIV_255: float =                           1.0 / 255.0
_1_DIV_360: float =                           1.0 / 360.0
_1_DIV_100: float =                           1.0 / 100.0
_1_DIV_1POINT055: float =                     1.0 / 1.055
_1_DIV_12POINT92: float =                     1.0 / 12.92
_1_DIV_95_047: float =                        1.0 / 95.047
_1_DIV_108_883: float =                       1.0 / 108.883
_1_DIV_116: float =                           1.0 / 116.0
_1_DIV_500: float =                           1.0 / 500.0
_1_DIV_200: float =                           1.0 / 200.0
_1_DIV_903_3: float =                         1.0 / 903.3
_1_THIRD: float =                             1.0 / 3.0
_1_SIXTH: float =                             1.0 / 6.0
_2_THIRDS: float =                            2.0 / 3.0

# fmt: on


def conversion_overload(overload_func, **overload_kwargs):
    """
    Decorator that provides the overloads for RGB conversions that result in
    float triplets (e.g., xyz, hls, hsv, etc.). The signature of the
    `overload_func` MUST be the following:

    def func_name(e, f, g):
        ...

    The base_implementation_func must have a three argument signature.
    """
    # no nans, no infs, and no signed zeros
    overload_kwargs["fastmath"] = overload_kwargs.pop("fastmath", {"nnan", "nsz", "ninf"})

    func_name = overload_func.__name__
    if func_name in ("_rgb_to_ind", "_ind_to_rgb"):
        # Most of the functions maintain the same array structure and pivot from
        # uint8 to float32 or visa versa. However, rgb_to_ind and ind_to_rgb,
        # remove and add dimensions to the outpput and pivot between uint8 and
        # uint32. So, those functions are handled separately
        return index_conversions_overloads(overload_func, **overload_kwargs)

    if "to_rgb" in func_name:
        itype = types.Float
        dtype = np.uint8

    elif "rgb_to" in func_name:
        itype = types.Integer
        dtype = np.float32

    else:
        itype = types.Float
        dtype = np.float32

    def inner(base_implementation_func):
        # make sure the base implementation can be run in a jitted context
        base_implementation_func = register_jitable(**overload_kwargs)(base_implementation_func)

        @overload(overload_func, **overload_kwargs)
        def base_impl(e, f, g):
            # implementation when all three arguments are itype
            if isinstance(e, itype) and isinstance(f, itype) and isinstance(g, itype):
                # ensure signature matches without forcing the
                # base_implementation_func's signature
                return lambda e, f, g: base_implementation_func(e, f, g)

            raise NumbaTypeError("NumbaTypeError")

        @overload(overload_func, **overload_kwargs)
        def tuple_impl(e, f, g):
            # implementation when first argument is a tuple
            if isinstance(e, types.UniTuple) and len(e) == 3:
                return lambda e, f, g: base_implementation_func(e[0], e[1], e[2])
            raise NumbaTypeError("NumbaTypeError")

        @overload(overload_func, **overload_kwargs)
        def array_impl(e, f, g):
            # implementation for arrays
            if isinstance(e, types.Array) and isinstance(e.dtype, itype):
                if e.ndim == 1:

                    def impl1D(e, f, g):
                        return np.array(
                            base_implementation_func(e[0], e[1], e[2]),
                            dtype=dtype,
                        )

                    return impl1D

                elif e.ndim == 2:

                    def impl2D(e, f, g):
                        height, _ = e.shape
                        out = np.zeros_like(e, dtype=dtype)
                        for i in prange(height):
                            out[i] = base_implementation_func(
                                e[i, 0],
                                e[i, 1],
                                e[i, 2],
                            )
                        return out

                    return impl2D
                elif e.ndim == 3:

                    def impl3D(e, f, g):
                        height, width, _ = e.shape
                        out = np.zeros_like(e, dtype=dtype)
                        for i in prange(height):
                            for j in range(width):
                                out[i, j] = base_implementation_func(
                                    e[i, j, 0],
                                    e[i, j, 1],
                                    e[i, j, 2],
                                )
                        return out

                    return impl3D
                elif e.ndim == 4:

                    def impl4D(e, f, g):
                        n_frames, height, width, _ = e.shape
                        out = np.zeros_like(e, dtype=dtype)
                        for i in prange(n_frames):
                            for j in range(height):
                                for k in range(width):
                                    out[i, j, k] = base_implementation_func(
                                        e[i, j, k, 0],
                                        e[i, j, k, 1],
                                        e[i, j, k, 2],
                                    )
                        return out

                    return impl4D

            raise NumbaTypeError("NumbaTypeError")

        return base_implementation_func

    return inner


def index_conversions_overloads(overload_func, **overload_kwargs):
    """
    The signature of the `overload_func` MUST be the following:

    def func_name(e, f, g):
        ...

    """
    func_name = overload_func.__name__
    if func_name == "_ind_to_rgb":
        itype = types.Integer
        fgtype = types.NoneType

    elif func_name == "_rgb_to_ind":
        itype = fgtype = types.Integer

    else:
        raise ValueError("Invalid function")

    def inner(base_implementation_func):
        # make sure the base implementation can be run in a jitted context
        base_implementation_func = register_jitable(**overload_kwargs)(base_implementation_func)

        @overload(overload_func, **overload_kwargs)
        def base_impl(e, f, g):
            # implementation when all three arguments are itype
            if isinstance(e, itype) and isinstance(f, fgtype) and isinstance(g, fgtype):
                # ensure signature matches without forcing the
                # base_implementation_func's signature
                return lambda e, f, g: base_implementation_func(e, f, g)
            raise NumbaTypeError("NumbaTypeError")

        # NO TUPLE IMPLEMENTATION
        if func_name == "_rgb_to_ind":

            @overload(overload_func, **overload_kwargs)
            def array_impl(e, f, g):
                # implementation for arrays
                if isinstance(e, types.Array) and isinstance(e.dtype, itype):
                    if 1 < e.ndim <= 4:

                        def impl(e, f, g):
                            return (
                                np.left_shift(e[..., 0], 16)
                                + np.left_shift(e[..., 1], 8)
                                + e[..., 2]
                            ).astype(np.uint32)

                        return impl

                    elif e.ndim == 1:

                        def impl(e, f, g):
                            return np.array(
                                [base_implementation_func(e[0], e[1], e[2])],
                                dtype=np.uint32,
                            )

                        return impl

                raise NumbaTypeError("NumbaTypeError")

        else:

            @overload(overload_func, **overload_kwargs)
            def array_impl(e, f, g):
                # implementation for arrays
                if isinstance(e, types.Array) and isinstance(e.dtype, itype) and (1 <= e.ndim <= 3):

                    def impl(e, f, g):
                        out = np.zeros(e.shape + (3,), dtype=np.uint8)
                        out[..., 0] = np.bitwise_and(np.right_shift(e, 16), 255)
                        out[..., 1] = np.bitwise_and(np.right_shift(e, 8), 255)
                        out[..., 2] = np.bitwise_and(e, 255)
                        return out

                    return impl

                raise NumbaTypeError("NumbaTypeError")

        return base_implementation_func

    return inner


################################################################################
### RGB --> HLS
################################################################################


def _rgb_to_hls(e, f, g):
    raise NotImplementedError


@conversion_overload(_rgb_to_hls, inline="always", cache=True)
def rgb_to_hls_(r, g, b):
    r *= _1_DIV_255
    g *= _1_DIV_255
    b *= _1_DIV_255
    minc, maxc = min(r, g, b), max(r, g, b)
    sumc = maxc + minc
    l = sumc / 2.0
    if minc == maxc:
        return 0.0, l * 100.0, 0.0

    rangec = maxc - minc
    if l <= 0.5:
        s = rangec / sumc
    else:
        s = rangec / (2.0 - sumc)
    rc = (maxc - r) / rangec
    gc = (maxc - g) / rangec
    bc = (maxc - b) / rangec
    if r == maxc:
        h = bc - gc
    elif g == maxc:
        h = 2.0 + rc - bc
    else:
        h = 4.0 + gc - rc
    h = (h / 6.0) % 1.0

    h *= 360.0
    l *= 100.0
    s *= 100.0
    return h, l, s


@njit(inline="always", cache=True)
def rgb_to_hls(r, g=None, b=None):
    return _rgb_to_hls(r, g, b)


################################################################################
### HLS --> RGB
################################################################################


def _hls_to_rgb(e, f, g):
    raise NotImplementedError


@conversion_overload(_hls_to_rgb, inline="always", cache=True)
def hls_to_rgb_(h, l, s):
    h *= _1_DIV_360
    l *= _1_DIV_100
    s *= _1_DIV_100
    if s == 0.0:
        l *= 255.0
        r = g = b = int(round(l))
        return r, g, b
    if l <= 0.5:
        m2 = l * (1.0 + s)
    else:
        m2 = l + s - (l * s)
    m1 = 2.0 * l - m2
    r, g, b = (_v(m1, m2, h + _1_THIRD), _v(m1, m2, h), _v(m1, m2, h - _1_THIRD))

    r *= 255.0
    g *= 255.0
    b *= 255.0
    return int(round(r)), int(round(g)), int(round(b))


@register_jitable(inline="always", cache=True)
def _v(m1, m2, hue):
    hue = hue % 1.0
    if hue < _1_SIXTH:
        return m1 + (m2 - m1) * hue * 6.0
    if hue < 0.5:
        return m2
    if hue < _2_THIRDS:
        return m1 + (m2 - m1) * (_2_THIRDS - hue) * 6.0
    return m1


@njit(inline="always", cache=True)
def hls_to_rgb(h, l=None, s=None):
    return _hls_to_rgb(h, l, s)


################################################################################
### RGB --> HSV
################################################################################


def _rgb_to_hsv(e, f, g):
    raise NotImplementedError


@conversion_overload(_rgb_to_hsv, inline="always", cache=True)
def rgb_to_hsv_(r, g, b):
    r *= _1_DIV_255
    g *= _1_DIV_255
    b *= _1_DIV_255
    minc, maxc = min(r, g, b), max(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v * 100.0

    rangec = maxc - minc
    s = rangec / maxc

    # rc = (maxc-r) / rangec
    # gc = (maxc-g) / rangec
    # bc = (maxc-b) / rangec
    denom = 1.0 / rangec
    if r == maxc:
        # h = bc-gc
        h = ((maxc - b) * denom) - ((maxc - g) * denom)
    elif g == maxc:
        # h = 2.0+rc-bc
        h = 2.0 + ((maxc - r) * denom) - ((maxc - b) * denom)
    else:
        # h = 4.0+gc-rc
        h = 4.0 + ((maxc - g) * denom) - ((maxc - r) * denom)

    h = (h / 6.0) % 1.0

    h *= 360.0
    s *= 100.0
    v *= 100.0
    return h, s, v


@njit(inline="always", cache=True)
def rgb_to_hsv(r, g=None, b=None):
    return _rgb_to_hsv(r, g, b)


################################################################################
### HSV --> RGB
################################################################################


def _hsv_to_rgb(e, f, g):
    raise NotImplementedError


@conversion_overload(_hsv_to_rgb, inline="always", cache=True)
def hsv_to_rgb_(h, s, v):
    h *= _1_DIV_360
    s *= _1_DIV_100
    v *= _1_DIV_100
    if s == 0.0:
        r = g = b = int(round(v * 100.0))
        return r, g, b

    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        r, g, b = v, t, p

    elif i == 1:
        r, g, b = q, v, p

    elif i == 2:
        r, g, b = p, v, t

    elif i == 3:
        r, g, b = p, q, v

    elif i == 4:
        r, g, b = t, p, v

    else:
        r, g, b = v, p, q

    r *= 255.0
    g *= 255.0
    b *= 255.0
    return int(round(r)), int(round(g)), int(round(b))


@njit(inline="always", cache=True)
def hsv_to_rgb(h, s=None, v=None):
    return _hsv_to_rgb(h, s, v)


################################################################################
### RGB --> XYZ
################################################################################


def _rgb_to_xyz(e, f, g):
    raise NotImplementedError


@conversion_overload(_rgb_to_xyz, inline="always", cache=True)
def rgb_to_xyz_(r, g, b):
    r = _pivot_rgb_to_xyz(r * _1_DIV_255)
    g = _pivot_rgb_to_xyz(g * _1_DIV_255)
    b = _pivot_rgb_to_xyz(b * _1_DIV_255)

    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    return x, y, z


@register_jitable([types.float64(types.float64)], cache=True)
def _pivot_rgb_to_xyz(value):
    if value <= 0.04045:
        value *= _1_DIV_12POINT92
    else:
        value = ((value + 0.055) * _1_DIV_1POINT055) ** 2.4
    return value * 100.0


@njit(inline="always", cache=True)
def rgb_to_xyz(r, g=None, b=None):
    """
    Convert tuple from the sRGB color space to the CIE XYZ color space.

    The XYZ output is determined using D65 illuminate with a 2° observer angle.
    https://en.wikipedia.org/wiki/Illuminant_D65

    The conversion matrix used was provided by Bruce Lindbloom:
    http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

    Formulas for conversion:
    http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    https://easyrgb.com/en/math.php

    Information about respective color space:
    sRGB (standard Red Green Blue): https://en.wikipedia.org/wiki/SRGB
    CIE XYZ: https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return _rgb_to_xyz(r, g, b)


################################################################################
### XYZ --> RGB
################################################################################


def _xyz_to_rgb(e, f, g):
    raise NotImplementedError


@conversion_overload(_xyz_to_rgb, inline="always", cache=True)
def xyz_to_rgb_(x, y, z):
    x *= _1_DIV_100
    y *= _1_DIV_100
    z *= _1_DIV_100

    r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    r = _pivot_xyz_to_rgb(r)
    g = _pivot_xyz_to_rgb(g)
    b = _pivot_xyz_to_rgb(b)

    return r, g, b


@register_jitable([types.uint8(types.float64)], cache=True)
def _pivot_xyz_to_rgb(value):
    if value <= 0.0031308:
        value *= 12.92
    else:
        value = ((value**0.4166666) * 1.055) - 0.055
    return int(round(value * 255.0))


@njit(inline="always", cache=True)
def xyz_to_rgb(x, y=None, z=None):
    """
    Convert tuple from the CIE XYZ color space to the sRGB color space.

    Conversion is based on that the XYZ input uses an the D65 illuminate with a 2° observer angle.
    https://en.wikipedia.org/wiki/Illuminant_D65

    The inverse conversion matrix used was provided by Bruce Lindbloom:
    http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html

    Formulas for conversion:
    http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    https://easyrgb.com/en/math.php

    Information about respective color space:
    sRGB (standard Red Green Blue): https://en.wikipedia.org/wiki/SRGB
    CIE XYZ: https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return _xyz_to_rgb(x, y, z)


################################################################################
### XYZ --> LAB
################################################################################


def _xyz_to_lab(e, f, g):
    raise NotImplementedError


@conversion_overload(_xyz_to_lab, inline="always", cache=True)
def xyz_to_lab_(x, y, z):
    x = _pivot_xyz_to_lab(x * _1_DIV_95_047)
    y = _pivot_xyz_to_lab(y * _1_DIV_100)
    z = _pivot_xyz_to_lab(z * _1_DIV_108_883)

    l = max(0.0, (116.0 * y) - 16.0)
    a = (x - y) * 500.0
    b = (y - z) * 200.0

    return l, a, b


@register_jitable([types.float64(types.float64)], cache=True)
def _pivot_xyz_to_lab(value):
    if value > 0.008856:
        value = value**_1_THIRD
    else:
        value = ((value * 903.3) + 16.0) * _1_DIV_116
    return value


@njit(inline="always", cache=True)
def xyz_to_lab(x, y=None, z=None):
    """
    Convert tuple from the CIE XYZ color space to the CIE L*a*b color space.

    Conversion is based on that the XYZ input uses an the D65 illuminate with a 2° observer angle.
    https://en.wikipedia.org/wiki/Illuminant_D65

    Formulas for conversion:
    https://en.wikipedia.org/wiki/CIELAB_color_space#CIELAB%E2%80%93CIEXYZ_conversions
    http://www.brucelindbloom.com/index.html?Eqn_XYZ_to_Lab.html
    https://easyrgb.com/en/math.php

    Information about respective color space:
    CIE L*a*b: https://en.wikipedia.org/wiki/Lab_color_space
    CIE XYZ: https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return _xyz_to_lab(x, y, z)


################################################################################
### LAB --> XYZ
################################################################################


def _lab_to_xyz(e, f, g):
    raise NotImplementedError


@conversion_overload(_lab_to_xyz, inline="always", cache=True)
def lab_to_xyz_(l, a, b):
    # Reminder: The y values is calculated first as it can be reused
    # for the calculation of x and z.
    y = (l + 16.0) * _1_DIV_116
    x = y + (a * _1_DIV_500)
    z = y - (b * _1_DIV_200)

    x3 = x**3
    z3 = z**3

    x = x3 if x3 > 0.008856 else ((x * 116.0) - 16.0) * _1_DIV_903_3
    y = (y**3) if l > 7.9996248 else l * _1_DIV_903_3
    z = z3 if z3 > 0.008856 else ((z * 116.0) - 16.0) * _1_DIV_903_3

    x *= 95.047
    y *= 100.000
    z *= 108.883

    return x, y, z


@njit(inline="always", cache=True)
def lab_to_xyz(l, a=None, b=None):
    """
    Convert tuple from the CIE L*a*b* color space to the CIE XYZ color space.

    The XYZ output is determined using D65 illuminate with a 2° observer angle.
    https://en.wikipedia.org/wiki/Illuminant_D65

    Formulas for conversion:
    https://en.wikipedia.org/wiki/CIELAB_color_space#CIELAB%E2%80%93CIEXYZ_conversions
    http://www.brucelindbloom.com/index.html?Eqn_RGB_to_XYZ.html
    https://easyrgb.com/en/math.php

    Information about respective color space:
    CIE L*a*b: https://en.wikipedia.org/wiki/Lab_color_space
    CIE XYZ: https://en.wikipedia.org/wiki/CIE_1931_color_space
    """
    return _lab_to_xyz(l, a, b)


################################################################################
### RGB --> LAB
################################################################################


def _rgb_to_lab(e, f, g):
    raise NotImplementedError


@conversion_overload(_rgb_to_lab, inline="always", cache=True)
def rgb_to_lab_(r, g, b):
    xyz = rgb_to_xyz_(r, g, b)
    return xyz_to_lab_(xyz[0], xyz[1], xyz[2])


@njit(inline="always", cache=True)
def rgb_to_lab(r, g=None, b=None):
    """
    Convert tuple from the sRGB color space to the CIE L*a*b* color space.
    Shorthand method for chaining sRGB => CIE XYZ => CIE L*a*b*.
    """
    return _rgb_to_lab(r, g, b)


################################################################################
### LAB --> RGB
################################################################################


def _lab_to_rgb(e, f, g):
    raise NotImplementedError


@conversion_overload(_lab_to_rgb, inline="always", cache=True)
def lab_to_rgb_(l, a, b):
    xyz = lab_to_xyz_(l, a, b)
    return xyz_to_rgb_(xyz[0], xyz[1], xyz[2])


@njit(inline="always", cache=True)
def lab_to_rgb(l, a=None, b=None):
    """
    Convert tuple from the CIE L*a*b* color space to the sRGB color space.
    Shorthand method for chaining CIE L*a*b* => CIE XYZ  => sRGB.
    """
    return _lab_to_rgb(l, a, b)


################################################################################
### RGB --> Indices
################################################################################


def _rgb_to_ind(e, f, g):
    raise NotImplementedError


@conversion_overload(_rgb_to_ind, inline="always", cache=True)
def rgb_to_ind_(r, g, b):
    return (r << 16) + (g << 8) + b


@njit(inline="always", cache=True)
def rgb_to_ind(r, g=None, b=None):
    """
    Uses bit shifting to get the index for an RGB triplet.
    """
    return _rgb_to_ind(r, g, b)


################################################################################
### Indices --> RGB
################################################################################


def _ind_to_rgb(e, f, g):
    raise NotImplementedError


@conversion_overload(_ind_to_rgb, inline="always", cache=True)
def ind_to_rgb_(e, f, g):
    return (e >> 16) & 255, (e >> 8) & 255, e & 255


@njit(inline="always", cache=True)
def ind_to_rgb(index):
    """
    Uses bit shifting to get the RGB triplet from an index.
    """
    return _ind_to_rgb(index, None, None)
