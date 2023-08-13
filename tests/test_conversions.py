import math
import typing as t
from random import randint

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import jcc.convert as jcc

ConversionFunc: t.TypeAlias = (
    t.Callable[[float, float, float], tuple[float, float, float]]
    | t.Callable[[int, int, int], tuple[float, float, float]]
    | t.Callable[[float, float, float], tuple[int, int, int]]
)
FloatTuple: t.TypeAlias = tuple[float, float, float] | tuple[int, int, int]


def all_close(
    func: ConversionFunc,
    value: FloatTuple,
    expected: FloatTuple,
    tol: float = 0.01,
) -> tuple[bool, float | None, float | None]:
    actual = func(*value)  # type: ignore
    for a, e in zip(actual, expected):
        if not math.isclose(a, e, abs_tol=tol):
            return False, e, a
    return True, None, None


@pytest.mark.parametrize(
    ["rgb", "xyz", "lab"],
    # fmt: off
    (
    ((255, 255, 255), (95.0470, 100.0000, 108.8830), (100.0000, 0, 0)),
    ((254, 254, 254), (94.2013,  99.1102, 107.9142), ( 99.6549, 0, 0)),
    ((230, 230, 230), (75.2105,  79.1298,  86.1589), ( 91.2930, 0, 0)),
    ((204, 204, 204), (57.3920,  60.3827,  65.7465), ( 82.0458, 0, 0)),
    ((179, 179, 179), (42.8458,  45.0786,  49.0829), ( 72.9436, 0, 0)),
    ((153, 153, 153), (30.2769,  31.8547,  34.6843), ( 63.2226, 0, 0)),
    ((128, 128, 128), (20.5169,  21.5861,  23.5035), ( 53.5850, 0, 0)),
    ((102, 102, 102), (12.6287,  13.2868,  14.4671), ( 43.1923, 0, 0)),
    (( 77,  77,  77), ( 7.0538,   7.4214,   8.0806), ( 32.7475, 0, 0)),
    (( 51,  51,  51), ( 3.1465,   3.3105,   3.6045), ( 21.2467, 0, 0)),
    (( 26,  26,  26), ( 0.9818,   1.0330,   1.1247), (  9.2632, 0, 0)),
    ((  1,   1,   1), ( 0.0288,   0.0304,   0.0330), (  0.2742, 0, 0)),
    ((  0,   0,   0), (      0,        0,        0), (       0, 0, 0)),
    )
    # fmt: on
)
def test_conversions_precision(rgb: FloatTuple, xyz: FloatTuple, lab: FloatTuple) -> None:
    is_good, expected, actual = all_close(jcc.rgb_to_xyz_, rgb, xyz)
    assert is_good, f"Expected: {expected} but got {actual} instead."
    is_good, expected, actual = all_close(jcc.xyz_to_rgb_, xyz, rgb)
    assert is_good, f"Expected: {expected} but got {actual} instead."
    is_good, expected, actual = all_close(jcc.xyz_to_lab_, xyz, lab)
    assert is_good, f"Expected: {expected} but got {actual} instead."
    is_good, expected, actual = all_close(jcc.lab_to_xyz_, lab, xyz)
    assert is_good, f"Expected: {expected} but got {actual} instead."
    is_good, expected, actual = all_close(jcc.rgb_to_lab_, rgb, lab)
    assert is_good, f"Expected: {expected} but got {actual} instead."
    is_good, expected, actual = all_close(jcc.lab_to_rgb_, lab, rgb)
    assert is_good, f"Expected: {expected} but got {actual} instead."


@pytest.mark.parametrize(
    "rgb",
    [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(20)],
)
@pytest.mark.parametrize(
    ["convert", "undo"],
    [
        (jcc.rgb_to_hls_, jcc.hls_to_rgb_),
        (jcc.rgb_to_xyz_, jcc.xyz_to_rgb_),
        (jcc.rgb_to_lab_, jcc.lab_to_rgb_),
        (jcc.rgb_to_hsv_, jcc.hsv_to_rgb_),
    ],
)
def test_conversion_reversible(
    rgb: tuple[int, int, int], convert: ConversionFunc, undo: ConversionFunc
):
    assert rgb == undo(*convert(*rgb))  # type: ignore


@pytest.mark.parametrize(
    "rgb",
    [(randint(0, 255), randint(0, 255), randint(0, 255)) for _ in range(20)],
)
@pytest.mark.parametrize(
    ["convert", "undo"],
    [
        (jcc.rgb_to_hls, jcc.hls_to_rgb),
        (jcc.rgb_to_xyz, jcc.xyz_to_rgb),
        (jcc.rgb_to_lab, jcc.lab_to_rgb),
        (jcc.rgb_to_hsv, jcc.hsv_to_rgb),
    ],
)
@pytest.mark.parametrize(
    "typ",
    [
        lambda rgb: rgb,  # as separate arguments
        lambda rgb: (rgb,),  # as a tuple
        lambda rgb: (np.array(rgb, dtype=np.uint8),),  # 1D array
        lambda rgb: (np.array(rgb, dtype=np.uint8).reshape(1, 3),),  # 2D array
        lambda rgb: (np.array(rgb, dtype=np.uint8).reshape(1, 1, 3),),  # 3D array
        lambda rgb: (np.array(rgb, dtype=np.uint8).reshape(1, 1, 1, 3),),  # 4D array
    ],
)
def test_conversion_types(
    rgb: t.Any,
    convert: t.Callable[..., t.Any],
    undo: t.Callable[..., t.Any],
    typ: t.Callable[..., t.Any],
):
    rgb = typ(rgb)
    converted = convert(*rgb)
    actual = undo(converted)
    expected = rgb

    if not isinstance(expected[0], int):
        expected = expected[0]

    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        assert_array_equal(actual, expected)
    else:
        assert expected == actual, f"Expected {expected}, but got {actual}"


@pytest.mark.parametrize(
    ["r", "g", "b"],
    [
        (255, 255, 255),
        (np.array([255, 255, 255], dtype=np.uint8), None, None),
        (np.array([[255, 255, 255]], dtype=np.uint8), None, None),
        (np.array([[[255, 255, 255]]], dtype=np.uint8), None, None),
    ],
)
def test_rgb_to_ind(r: t.Any, g: t.Any, b: t.Any):
    expected = (256**3) - 1

    actual = jcc.rgb_to_ind(r, g, b)  # type: ignore
    if isinstance(actual, np.ndarray):
        actual = actual[..., 0]  # type: ignore
        assert (actual == expected).all()
    else:
        assert actual == expected
