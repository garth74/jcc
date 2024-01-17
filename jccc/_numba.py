import os as _os

from numba import njit as njit  # type: ignore
from numba import prange as prange
from numba.core import types as types  # type: ignore
from numba.extending import register_jitable as register_jitable  # type: ignore

# https://numba.readthedocs.io/en/latest/reference/deprecation.html#deprecation-of-old-style-numba-captured-errors
_os.environ["NUMBA_CAPTURED_ERRORS"] = "new_style"
