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
    InsertPlanarFromDistribution,
)
from .lib import (
    PBTwoPlates,
    PBTwoPlates2
)


__all__ = [
    "InsertPlanar",
    "InsertCylinder",
    "InsertSphere",
    "SolvatePlanar",
    "SolvateCylinder",
    "InsertPlanarFromDistribution",
    "models",
    "PBTwoPlates",
    "PBTwoPlates2",
]

from . import _version


__version__ = _version.get_versions()["version"]
