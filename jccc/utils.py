from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jccc._numba import njit, prange
from jccc.convert import rgb_to_lab
from jccc.diff import delta_e_cie2000

Float16Array: t.TypeAlias = npt.NDArray[np.float16]
Float32Array: t.TypeAlias = npt.NDArray[np.float32]
Float64Array: t.TypeAlias = npt.NDArray[np.float64]
Uint8Array: t.TypeAlias = npt.NDArray[np.uint8]
Uint16Array: t.TypeAlias = npt.NDArray[np.uint16]
Uint32Array: t.TypeAlias = npt.NDArray[np.uint32]
AnyArray: t.TypeAlias = npt.NDArray[t.Any]
IntpArray: t.TypeAlias = npt.NDArray[np.intp]
BoolArray: t.TypeAlias = npt.NDArray[np.bool_]
StrArray: t.TypeAlias = npt.NDArray[np.str_]


# ============================================================================ #
# Access package data


DATA_DIR = Path(__file__).parent / "data"


def get_data_path(filename: str, *, strict: bool = False) -> Path:
    """
    Construct a filepath to the data directory. If `strict` is True, a missing
    file with throw an error.
    """
    path = DATA_DIR / filename
    if strict and not path.exists():
        err = FileNotFoundError("Failed to find internal package file.")
        err.add_note("This is an internal error. Please report this.")
        raise err

    return path


# ============================================================================ #
# Create a lookup table


@njit(parallel=True)
def create_lookup_table(arr_lab: Float64Array, all_lab: Float64Array) -> Uint16Array:
    arr_lab_N = len(arr_lab)
    out = np.zeros(256**3, dtype=np.uint16)
    for i in prange(256**3):
        lab1 = all_lab[i]
        deltaes = np.zeros(arr_lab_N, dtype=np.float32)
        for j in prange(arr_lab_N):
            deltaes[j] = delta_e_cie2000(
                lab1[0],
                lab1[1],
                lab1[2],
                arr_lab[j][0],
                arr_lab[j][1],
                arr_lab[j][2],
            )
        out[i] = np.argmin(deltaes)

    return out


@njit(inline="always", cache=True)
def get_all_rgbs() -> Uint8Array:
    """
    Returns an array of all RGB triplets.
    """
    rgb = np.empty((256**3, 3), dtype=np.uint8)
    arr = np.arange(256, dtype=np.uint8).repeat(256**2)
    rgb[:, 0] = arr
    rgb[:, 1] = arr.reshape((-1, 256)).T.ravel()
    rgb[:, 2] = arr.reshape((-1, 256**2)).T.ravel()
    return rgb


@njit(cache=True)
def get_all_lab() -> Float64Array:
    """
    Returns a numpy array of the RGB colors in the LAB colorspace.
    """
    return rgb_to_lab(get_all_rgbs())
