"""
nonmaicos

A MD manipulation library.
"""

__authors__ = "MAICoS Developer Team"


from . import models
from .insert import (
    InsertCylinder,
    InsertPlanar,
    InsertSphere,
    SolvateCylinder,
    SolvatePlanar,
)


__all__ = [
    "InsertPlanar",
    "InsertCylinder",
    "InsertSphere",
    "SolvatePlanar",
    "SolvateCylinder",
    "models",
]

from . import _version


__version__ = _version.get_versions()["version"]
