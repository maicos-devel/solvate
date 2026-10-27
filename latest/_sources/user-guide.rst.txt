User guide
==========

This guide goes a bit deeper than :doc:`get-started`. It covers the
geometric primitives available for insertion, the trade-off between the
``Insert*`` and ``Solvate*`` family, the built-in water models, and the
knobs you can turn when a difficult system refuses to pack.

Insertion regions
-----------------

solvate provides three insertion regions. All of them accept the same
``target``, ``projectile``, and ``n`` arguments and differ only in the
shape of the volume they fill.

Rectangular boxes
~~~~~~~~~~~~~~~~~

:func:`~solvate.InsertPlanar` fills an axis-aligned box. By default the
box is the full simulation cell of the target; use ``xmin/xmax``,
``ymin/ymax``, ``zmin/zmax`` to carve out a sub-region — for example a
solvent slab on top of a surface:

.. code-block:: python

    # Insert water only between z = 20 Å and z = 40 Å
    slab = solvate.InsertPlanar(
        surface, solvate.models.spce(), n=500, zmin=20.0, zmax=40.0,
    )

Cylinders
~~~~~~~~~

:func:`~solvate.InsertCylinder` fills a cylinder. ``dim`` selects the axis
(0 = x, 1 = y, 2 = z; default ``dim=2``), ``radius`` controls the cross
section, and ``min/max`` are the lower and upper bounds along the cylinder
axis. ``pos`` sets the cylinder centre; when omitted it falls on the centre
of geometry of the target (or the centre of the box for an empty target).

.. code-block:: python

    # Fill a pore of radius 8 Å aligned with z, between z = 0 and z = 50 Å
    pore = solvate.InsertCylinder(
        target, solvate.models.spce(), n=200, radius=8.0, max=50.0,
    )

Spheres
~~~~~~~

:func:`~solvate.InsertSphere` fills a sphere centred at ``pos`` (default:
target centre of geometry / box centre) with the given ``radius``.

.. code-block:: python

    # Build a 20 Å water droplet
    droplet = solvate.InsertSphere(
        box, solvate.models.spce(), n=1000, radius=20.0,
    )

``Insert*`` versus ``Solvate*``
-------------------------------

The ``Insert*`` functions place projectiles one by one and re-check for
overlaps with every new candidate. They are the right choice for small
numbers of insertions (a few hundred at most) or when you need full control
over the placement order.

The ``Solvate*`` family — currently :func:`~solvate.SolvatePlanar` and
:func:`~solvate.SolvateCylinder` — uses a much faster strategy:

1. Pack a small "seed" patch of solvent until it is saturated.
2. Tile that patch across the target region.
3. Remove projectiles that clash with the target in a single batched
   neighbour search.

This is the recommended approach whenever you want to solvate a target
with many thousands of solvent molecules. It is also the way to specify a
``density`` instead of an explicit ``n``.

Built-in water models
---------------------

The :mod:`solvate.models` module gives you ready-to-use water models as
:class:`MDAnalysis Universes <MDAnalysis.core.universe.Universe>`:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Function
     - Model
     - Notes
   * - :func:`~solvate.models.spce`
     - SPC/E
     - 3-site, θ = 109.47°
   * - :func:`~solvate.models.tip3p`
     - TIP3P
     - 3-site, θ = 104.52°
   * - :func:`~solvate.models.tip4p_epsilon`
     - TIP4P/ε
     - 4-site (extra massless M site)

Need a custom 3- or 4-site model? Build it directly with
:func:`~solvate.models.type_a` (3-site) or :func:`~solvate.models.type_c`
(4-site), passing your bond length, partial charges, and HOH angle.

.. code-block:: python

    import numpy as np
    import solvate

    # A toy 3-site model with a 100° HOH angle
    custom = solvate.models.type_a(
        l_1=1.0, q_O=-0.8, q_H=0.4, theta=np.deg2rad(100.0),
    )

To start from a truly empty system (no atoms, just box dimensions), use
:func:`~solvate.models.empty`.

Tuning the solver
-----------------

When the target is densely packed, or when you ask for many projectiles in
a small region, placement can become difficult. A few knobs help:

``distance`` (default ``1.25`` Å)
    Minimum distance between a projectile and any existing atom. Lowering
    it allows tighter packing at the risk of unphysical contacts.

``tries`` (default ``1000``)
    Maximum random placement attempts per projectile before
    :func:`~solvate.InsertPlanar` (or its siblings) raises a
    :class:`RuntimeError`. Increase it when packing is tight.

``fudge_factor`` (``Solvate*`` only, default ``1.0``)
    Multiplier applied to the seed-patch population. The ``Solvate*``
    functions automatically increase it and retry when not enough
    projectiles survived the overlap pruning, so you rarely need to set
    it by hand.

``density`` versus ``n``
    Pass ``density`` (in molecules / Å³) to let solvate compute the number
    of projectiles from the volume of the insertion region. Pass ``n``
    when you need an exact molecule count.

Tips and gotchas
----------------

- **Set the simulation cell before inserting.** Random positions are drawn
  from the cell dimensions, so a missing or zero-sized box leads to
  surprising results. Use :func:`solvate.models.empty` to construct an
  empty Universe with a well-defined box.
- **Prefer ``Solvate*`` for large boxes.** A 50 × 50 × 50 Å³ water box
  with :func:`~solvate.InsertPlanar` will take a long time;
  :func:`~solvate.SolvatePlanar` does it in seconds.
- **Write whatever format you need.** The returned object is a regular
  MDAnalysis Universe — ``universe.atoms.write("foo.gro")``,
  ``"foo.xyz"``, ``"foo.pdb"`` all work.
