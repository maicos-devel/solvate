Getting started
===============

This page walks you through installing solvate and building your first
solvated system in a few lines of Python.

Installation
------------

Install the latest release from PyPI:

.. code-block:: bash

    pip install solvate

Or install the development version directly from GitHub:

.. code-block:: bash

    git clone https://github.com/maicos-devel/solvate.git
    cd solvate
    pip install -e .

solvate requires Python ≥ 3.10 and pulls in :mod:`MDAnalysis` automatically.

Core concepts
-------------

Every solvate call works with two
:class:`MDAnalysis Universes <MDAnalysis.core.universe.Universe>`:

- the **target** — the system you want to add molecules into (a slab,
  pore, droplet, or an empty box);
- the **projectile** — the molecule that gets inserted, possibly many times
  (a water molecule, an ion, etc.).

solvate returns a new :class:`~MDAnalysis.core.universe.Universe` containing
the target *plus* the inserted projectiles. You can write it to any format
MDAnalysis supports.

Inserting a single molecule
---------------------------

The simplest workflow loads a target and a projectile from files and drops
one copy of the projectile somewhere inside the target's box:

.. code-block:: python

    import MDAnalysis as mda
    import solvate

    target = mda.Universe("target.pdb")
    projectile = mda.Universe("projectile.pdb")

    universe = solvate.InsertPlanar(target, projectile, n=1)
    universe.atoms.write("solvated_system.pdb")

:func:`~solvate.InsertPlanar` places projectiles at random positions and
orientations inside a rectangular region, retrying when a placement would
overlap an existing atom.

Building a water box
--------------------

solvate ships with the most common water models, so building a water box
does not require any input files:

.. code-block:: python

    import solvate

    # A 30 Å cubic, empty target box
    box_length = 30.0
    box = solvate.models.empty([box_length] * 3 + [90, 90, 90])

    # SPC/E water at ~1 g/cm^3 corresponds to roughly 0.033 molecules / Å^3
    target_density = 0.033
    n_waters = int(target_density * box_length**3)

    water_box = solvate.InsertPlanar(box, solvate.models.spce(), n=n_waters)
    water_box.atoms.write("water_box.pdb")

If you do not need a specific molecule count, ask for a **density** directly
and let solvate compute ``n`` for you. For boxes larger than a few hundred
molecules use the optimized :func:`~solvate.SolvatePlanar` instead of
:func:`~solvate.InsertPlanar`:

.. code-block:: python

    water_box = solvate.SolvatePlanar(box, solvate.models.spce(), density=0.033)
    water_box.atoms.write("water_box.pdb")

:func:`~solvate.SolvatePlanar` builds a small saturated patch of solvent and
tiles it across the target volume, so it scales to tens of thousands of
molecules in seconds.

Solvating a target
------------------

The same call works when the target already contains atoms — for example a
membrane, a protein, or a slit pore. solvate inserts projectiles only where
they do not overlap the existing atoms:

.. code-block:: python

    membrane = mda.Universe("membrane.pdb")
    solvated = solvate.SolvatePlanar(membrane, solvate.models.spce(), density=0.033)
    solvated.atoms.write("membrane_solvated.pdb")

By default the insertion region is the full simulation box of the target;
restrict it via the ``xmin/xmax``, ``ymin/ymax``, and ``zmin/zmax``
arguments to solvate, say, only a slab above a surface.

Where to go next
----------------

- :doc:`user-guide` covers the different insertion regions, available water
  models, and how to tune the solvation parameters.
- :doc:`api` lists every public function with its full signature.
