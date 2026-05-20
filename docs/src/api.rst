API reference
=============

This page lists every public function exported by solvate, grouped by
module.

Insertion and solvation
-----------------------

These are the functions for placing molecules in a target
Universe. See :doc:`user-guide` for guidance on which one to pick.

.. currentmodule:: solvate

.. autofunction:: InsertPlanar
.. autofunction:: InsertCylinder
.. autofunction:: InsertSphere
.. autofunction:: SolvatePlanar
.. autofunction:: SolvateCylinder

Water and small-molecule models
-------------------------------

Pre-built and parametric models, all returning a fresh
:class:`MDAnalysis Universe <MDAnalysis.core.universe.Universe>`.

.. currentmodule:: solvate.models

.. autofunction:: empty
.. autofunction:: spce
.. autofunction:: tip3p
.. autofunction:: tip4p_epsilon
.. autofunction:: type_a
.. autofunction:: type_c
