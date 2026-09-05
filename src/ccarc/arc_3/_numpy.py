"""numpy, imported the first time something actually uses it.

Importing numpy dominates the cost of a cold ``import ccarc.arc_3``, and a
solver pays that on **every** Bash call. A call that asks only what was *done*
-- levels, actions, scores, resets -- never builds an array, so it should not
pay for numpy at all.

The proxy removes itself. On the first attribute lookup it imports numpy, binds
the real module into the namespaces that asked for it, and returns the
attribute. Every later ``np.something`` in those modules is an ordinary global
lookup against the real module, so the indirection is paid once rather than per
board -- which matters, because `_grids` touches one array per frame and a
trace holds many thousands of them.
"""
from __future__ import annotations

from typing import Any


class _LazyNumpy:
    """Stands in for ``numpy`` until the first attribute is asked for."""

    __slots__ = ("_namespaces",)

    def __init__(self, *namespaces: dict) -> None:
        self._namespaces = namespaces

    def __getattr__(self, name: str) -> Any:
        import numpy

        for namespace in self._namespaces:
            if namespace.get("np") is self:
                namespace["np"] = numpy
        return getattr(numpy, name)


def lazy_numpy(*namespaces: dict) -> Any:
    """A stand-in for ``numpy`` that replaces itself in ``namespaces`` on use."""
    return _LazyNumpy(*namespaces)
