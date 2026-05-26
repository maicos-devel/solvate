solvate
=======

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.20395882.svg
   :target: https://doi.org/10.5281/zenodo.20395882
   :alt: DOI

**solvate** is a lightweight Python package for quickly building initial
structures for molecular dynamics simulations. It is built on top of
:class:`MDAnalysis <MDAnalysis.core.universe.Universe>`, so it interoperates
cleanly with the rest of the MDAnalysis ecosystem and supports every file
format MDAnalysis can read or write.

It is inspired by tools such as GROMACS ``solvate``, but exposes a small,
scriptable Python API that is easy to compose with your own pre-processing
pipeline.

Highlights
----------

- **Geometric insertion** — fill rectangular, cylindrical, or spherical
  regions with arbitrary molecules.
- **Fast solvation** — the ``Solvate*`` variants use spatial tiling to
  populate large boxes orders of magnitude faster than naive insertion.
- **Specify** ``n`` **or density** — ask for an exact number of molecules or
  a target number density; solvate figures out the rest.
- **Built-in water models** — SPC/E, TIP3P, and TIP4P/ε are included; custom
  3- and 4-site models are one function call away.
- **MDAnalysis-native** — inputs and outputs are
  :class:`~MDAnalysis.core.universe.Universe` objects, so writing to any
  supported format is a one-liner.

A 30-second example
-------------------

.. code-block:: python

    import solvate

    # A 30 Å cubic, empty box
    box = solvate.models.empty([30, 30, 30, 90, 90, 90])

    # Fill it with SPC/E water at ~1 g/cm^3 (≈ 0.033 molecules / Å³)
    water_box = solvate.SolvatePlanar(box, solvate.models.spce(), density=0.033)

    water_box.atoms.write("water_box.pdb")

.. toctree::
   :maxdepth: 2
   :caption: User documentation

   get-started
   user-guide
   api

Citing solvate
--------------

If you use solvate in your research, please cite it via its Zenodo record:
`10.5281/zenodo.20395882 <https://doi.org/10.5281/zenodo.20395882>`_.
