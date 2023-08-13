import typing as t
from math import copysign, log2
from pathlib import Path

import numpy as np

from jcc.palettes._base import Palette
from jcc.palettes.types import *

x11 = Palette(Path(__file__).parent / "data/x11.csv")
"""Represents the X11 color palette."""

_PALETTES: dict[str, Palette] = {"x11": x11}
"Used to get the palette by name."
_ETS = 1e-15
"Used in color_entropy function to make sure we aren't taking the log2 of 0."

PaletteName: t.TypeAlias = t.Literal["x11"]


def quantize_with_palette(img: Uint8Array, palette: PaletteName | Palette) -> Uint8Array:
    """Reduce the RGB values in an image to the nearest values in a color
    palette.
    """
    return _get_palette(palette).convert_to_rgbs(img)


def color_histogram(img: Uint8Array, palette: PaletteName | Palette) -> AnyArray:
    """Creates a histogram of the colors in `img` using `palette`. The output is
    a structured array with all the color palette data plus the `count`. The
    image is converted to the palette indices, so the image doesn't need to be
    quantized beforehand.
    """
    pal = _get_palette(palette)
    indices = pal.convert_to_indices(img)
    counts = np.bincount(indices.ravel(), minlength=pal.n_colors)

    # create the structured array with the data
    dtype = pal.DTYPE.copy()
    dtype.append(("count", "i8"))

    palette_data = pal.data
    data: AnyArray = np.zeros(palette_data.shape, dtype=dtype)

    # set the values
    for key in t.cast(tuple[str, ...], palette_data.dtype.names):
        data[key] = palette_data[key]

    data["count"] = counts

    return data


def group_color_histogram(data: AnyArray) -> AnyArray:
    """Groups the `color_histogram` output by the color groups (e.g., blue, red,
    etc.). The output is a structured array with the group name (i.e., `color`)
    and the number of pixels in that group (i.e., `count`).
    """
    colors: list[str] = np.unique(data["group"]).tolist()
    grouped_data = np.zeros(len(colors), dtype=[("color", "U20"), ("count", "i8")])
    for index, color in enumerate(colors):
        grouped_data[index]["color"] = color

        subset = data[data["group"] == color]
        grouped_data[index]["count"] = np.sum(subset["count"])
    return grouped_data


def color_entropy(data: AnyArray) -> float:
    """Calculates the metric entropy based on a color histogram. Note that
    black, white, and gray are merged into a single group `grayscale`. This is
    to handle grayscale images which contain only black, white and shades of
    gray.
    """
    total_count = data["count"].sum()
    # calculate a single probability for the grayscale colors
    is_grayscale = np.isin(data["color"], ("black", "white", "gray"))
    probs: list[float] = [float(data[is_grayscale]["count"].sum() / total_count)]
    probs.extend((data[~is_grayscale]["count"] / total_count).tolist())
    value = -sum([p * log2(p + _ETS) for p in probs]) / len(probs)
    # The value can be negative if it's really close to zero
    value *= copysign(1.0, value)
    return value


def _get_palette(palette: PaletteName | Palette) -> Palette:
    """Helper function used to handle palette argument."""
    if isinstance(palette, Palette):
        return palette
    else:
        try:
            return _PALETTES[palette]
        except KeyError:
            raise NotImplementedError(f"Unsupported palette: {palette}") from None
