from __future__ import annotations

import functools
import typing as t
from pathlib import Path

import numpy as np

from jcc._numba import njit, prange
from jcc.convert import rgb_to_ind, rgb_to_lab
from jcc.diff import delta_e_cie2000
from jcc.palettes.types import *


def _iterlines(path: Path | str, rgb2hex: t.Callable[[int], str] = "#{:02x}{:02x}{:02x}".format):
    with open(path, "r") as handle:
        for line in handle:
            parts = line.strip().split(",")
            parts.append(rgb2hex(*map(int, parts[-3:])))
            yield ",".join(parts)


class Palette:
    """
    Represents a palette of colors. Note that palettes cannot have more than
    2**16 colors.
    """

    COLORS = [
        "black",
        "blue",
        "brown",
        "cyan",
        "gray",
        "green",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    ]

    DTYPE: list[tuple[str, str]] = [
        ("group", "U20"),
        ("name", "U50"),
        ("r", "u1"),
        ("g", "u1"),
        ("b", "u1"),
        ("hex", "U7"),
    ]

    def __init__(self, path: str | Path) -> None:
        self.data_path = Path(path)
        """Path to a CSV file containing the palette data."""
        self.cached_data_path = self.data_path.with_suffix(".npy")
        """Path to the cached lookup table if it has been created."""

    @property
    def n_colors(self) -> int:
        """
        Returns the number of colors in the palette.
        """
        return self.data.shape[0]

    @functools.cached_property
    def lookup_table(self) -> Uint16Array:
        """
        Returns a 256**3 len numpy array with the indices for each RGB value.
        """
        if self.cached_data_path.exists():
            return np.load(self.cached_data_path, allow_pickle=True)
        raise FileNotFoundError(
            """This look up table has not been created yet. First call the
            `build` method to create the lookup table."""
        )

    @functools.cached_property
    def data(self) -> AnyArray:
        """
        This should be a structured array with, at minimum 4 columns. The first
        column contains the color group name (e.g., black, blue, etc.). The next
        3 columns must have the RGB values. The entire array must be sorted by
        color group, then by the rgb values.
        """
        lines = _iterlines(self.data_path)
        return np.loadtxt(lines, dtype=self.DTYPE, delimiter=",")

    def convert_to_rgbs(self, arr: Uint8Array) -> Uint8Array:
        """
        Converts an array of RGB values in corresponding palette RGBs.
        """
        return self.rgbs[self.convert_to_indices(arr)]

    def convert_to_indices(self, arr: Uint8Array) -> Uint16Array:
        """
        Converts an array of RGB values to the corresponding palette indices.
        This is useful for getting other palette information or just reducing
        memory usage.
        """
        return self.lookup_table[rgb_to_ind(arr)]

    def clear_cache(self) -> None:
        """
        The `lookup_table` property is a cached_property, so this just clears
        the cache.
        """
        del self.lookup_table

    @property
    def rgbs(self) -> Uint8Array:
        """
        Returns an array of the RGB values for the palette.
        """
        return np.column_stack(
            (
                self.data["r"],
                self.data["g"],
                self.data["b"],
            ),
        ).astype(np.uint8)

    @property
    def groups(self) -> StrArray:
        return self.data["group"]

    def build(self) -> None:
        arr = _create_lookup_table(rgb_to_lab(self.rgbs), _load_lab_array())
        np.save(self.cached_data_path, arr, allow_pickle=True)


@njit(parallel=True)
def _create_lookup_table(arr_lab: Float64Array, all_lab: Float64Array) -> Uint16Array:
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
def _get_all_rgbs() -> Uint8Array:
    """Returns an array of all RGB triplets."""
    rgb = np.empty((256**3, 3), dtype=np.uint8)
    arr = np.arange(256, dtype=np.uint8).repeat(256**2)
    rgb[:, 0] = arr
    rgb[:, 1] = arr.reshape((-1, 256)).T.ravel()
    rgb[:, 2] = arr.reshape((-1, 256**2)).T.ravel()
    return rgb


@njit(cache=True)
def __create_lab_array() -> Float64Array:
    return rgb_to_lab(_get_all_rgbs())


def _load_lab_array() -> Float64Array:
    path = Path(__file__).parent / "data/lab.npy"
    if not path.exists():
        lab_array = __create_lab_array()
        np.save(path, lab_array, allow_pickle=True)
        return lab_array
    return np.load(path, allow_pickle=True)
