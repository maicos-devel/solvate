"""solvate: An MD manipulation library."""

from importlib.metadata import metadata

from . import models
from ._version import __version__  # noqa: F401
from .insert import (
    InsertCylinder,
    InsertPlanar,
    InsertSphere,
    SolvateCylinder,
    SolvatePlanar,
)

_meta = metadata("solvate")
__authors__ = _meta["Author"]

__all__ = [
    "InsertPlanar",
    "InsertCylinder",
    "InsertSphere",
    "SolvatePlanar",
    "SolvateCylinder",
    "models",
]
