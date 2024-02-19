"""
nonmaicos

A MD manipulation library.
"""

__author__ = "Henrik Jaeger"
__credits__ = "ICP, Uni Stuttgart"


from .insert import (
    InsertPlanar,
    InsertCylinder,
    InsertSphere,
    SolvatePlanar,
    SolvateCylinder,
)
from . import models

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
