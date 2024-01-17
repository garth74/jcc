import typing as t

from numba import prange as prange
from numba.core import types as types

_P = t.ParamSpec("_P")
_R = t.TypeVar("_R")

@t.overload
def njit(*args: t.Callable[_P, _R]) -> t.Callable[_P, _R]: ...  # type: ignore[misc]
@t.overload
def njit(**kwargs: t.Any) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]]: ...
@t.overload
def njit(
    *args: list[t.Any], **kwargs: t.Any
) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]]: ...
@t.overload
def njit(
    *args: t.Callable[_P, _R] | list[t.Any], **kwargs: t.Any
) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]] | t.Callable[_P, _R]: ...
@t.overload
def register_jitable(*args: t.Callable[_P, _R]) -> t.Callable[_P, _R]: ...  # type: ignore[misc]
@t.overload
def register_jitable(**kwargs: t.Any) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]]: ...
@t.overload
def register_jitable(
    *args: list[t.Any], **kwargs: t.Any
) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]]: ...
@t.overload
def register_jitable(
    *args: t.Callable[_P, _R] | list[t.Any], **kwargs: t.Any
) -> t.Callable[[t.Callable[_P, _R]], t.Callable[_P, _R]] | t.Callable[_P, _R]: ...
