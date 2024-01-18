from __future__ import annotations

import typing as t
from pathlib import Path

import numpy as np
import numpy.typing as npt

from jccc.convert import rgb_to_ind, rgb_to_lab
from jccc.utils import create_lookup_table, get_all_lab, get_data_path

Float64Array: t.TypeAlias = npt.NDArray[np.float64]
Uint8Array: t.TypeAlias = npt.NDArray[np.uint8]
Uint16Array: t.TypeAlias = npt.NDArray[np.uint16]
AnyArray: t.TypeAlias = npt.NDArray[t.Any]
StrArray: t.TypeAlias = npt.NDArray[np.str_]

Dtypeish: t.TypeAlias = list[tuple[str, str]]
NumpyLoadable: t.TypeAlias = Path | str | t.Iterable[str]


def _load_data(data: NumpyLoadable | None, dtype: Dtypeish) -> AnyArray:
    """Load the data using numpy."""
    # Validate the dtype
    expects = {"group", "name", "r", "g", "b"}
    actual = {elem[0] for elem in dtype}
    if not expects.issubset(actual):
        err = ValueError()
        err.add_note(f"Missing required columns in `dtype`:\n{sorted(expects - actual):!r}")
        raise err

    # Validate the data
    if data is None:
        raise ValueError("An argument to must be provided for either `file` or `lines")

    data = np.loadtxt(data, dtype=dtype, delimiter=",")
    data = np.sort(data, order=["group", "r", "g", "b"])
    return data


# ============================================================================ #
# BasePalette
# ============================================================================ #


class _BasePalette:
    """
    Base class used by all color palettes.
    """

    def __init__(self, name: t.LiteralString, data: NumpyLoadable | None, dtype: Dtypeish) -> None:
        self._name = name
        self._data = _load_data(data, dtype)
        self._lookup_table: Uint16Array | None = None

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

    @property
    def lookup_table_path(self) -> Path:
        """
        Return the path to the corresponding lookup table.
        """
        return getattr(self, "_lookup_table_path", Path(f"{self._name}.npz").absolute())

    @property
    def lookup_table(self) -> Uint16Array:
        """
        Returns the lookup table.
        """
        if self._lookup_table is None:
            path = self.lookup_table_path
            if not path.exists():
                err = FileNotFoundError(path)
                err.add_note("Call `write_lookup_table` first.")
                raise err

            self._lookup_table = np.load(path)[self._name]

        return self._lookup_table

    @lookup_table.setter
    def lookup_table(self, value: Uint16Array | None) -> None:
        self._lookup_table = value

    def write_lookup_table(self, *, overwrite: bool = False, strict: bool = True) -> None:
        """
        Write a lookup table to a path using the working directory and the name
        of the palette.
        """
        path = self.lookup_table_path
        if not overwrite and path.exists():
            if not strict:
                return

            raise FileExistsError(path)

        all_lab = get_all_lab()
        arr_lab = rgb_to_lab(self.rgbs)
        kwargs = {self._name: create_lookup_table(arr_lab, all_lab)}
        np.savez_compressed(path, **kwargs)

    @property
    def data(self) -> AnyArray:
        """
        Returns the loaded palette data.
        """
        return self._data

    @property
    def rgbs(self) -> Uint8Array:
        """
        Return the `r`, `g`, `b` columns as a stacked array.
        """
        return np.column_stack(tuple(self.data[k] for k in "rgb")).astype(np.uint8)

    @property
    def groups(self) -> StrArray:
        """
        Return the `group` column.
        """
        return self.data["group"]

    @property
    def names(self) -> StrArray:
        """
        Return the `name` column.
        """
        return self.data["name"]

    @property
    def n_colors(self) -> int:
        """
        Returns the number of colors in the palette.
        """
        return self.data.shape[0]


# ============================================================================ #
# Palette


class Palette(_BasePalette):
    """
    This class represents a color palette that can be used for color
    conversions.
    """


# ============================================================================ #
# CSS4 Palette


class Css4Palette(Palette):
    """
    Represents the CSS4 color palette.
    """

    def __init__(self) -> None:
        super().__init__(
            "css4",
            get_data_path("palettes/css4.csv", strict=True),
            [
                ("group", "U20"),
                ("name", "U50"),
                ("r", "u1"),
                ("g", "u1"),
                ("b", "u1"),
            ],
        )
        self._lookup_table_path = get_data_path("palettes/css4.npz")


if __name__ == "__main__":
    palettes = [Css4Palette]
    # construct the palette lookup tables
    for klass in palettes:
        palette = klass()
        print(f"Creating {klass.__name__} lookup table")
        palette.write_lookup_table(strict=False)
