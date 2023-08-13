import typing as t

import numpy as np
import numpy.typing as npt
from numpy.typing import DTypeLike as DTypeLike

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
