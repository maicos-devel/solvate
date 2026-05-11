Getting started
===============

Welcome to the solvate documentation! This guide will help you get started with using solvate for your projects.

Installation
------------

To install solvate, you can use pip:

.. code-block:: bash

    pip install solvate

or clone the repository from GitHub:

.. code-block:: bash

    git clone https://github.com/yourusername/solvate.git
    cd solvate
    pip install .

Basic Usage
-----------

Solvate allows you to insert `Projectiles` into `Targets`, with both being :class:`MDAnalysis <MDAnalysis>` Universe objects. Here's a simple example:

.. code-block:: python

    import MDAnalysis as mda
    import solvate

    target = mda.Universe('target.pdb')
    projectile = mda.Universe('projectile.pdb')

    universe = solvate.Insert(target=target, projectile=projectile, n=1)

    solvated_system.atoms.write('solvated_system.pdb')

Working with water
------------------

Solvate comes with built-in support for common water models like TIP3P and SPC/E. You can easily create a water box:

.. code-block:: python

    import solvate

    box_lengths = 10.0  # Define box lengths
    box_size = [box_lengths, box_lengths, box_lengths, 90, 90, 90]  # Define box dimensions
    target_density = 0.033 # 33 Water molecules per Å^3 ~ 1 g/cm^3

    spce = solvate.models.spce()

    empty_box = solvate.models.empty(box_size)

    water_box = solvate.InsertPlanar(empty_box, spce, n=target_density * box_lengths**3)
    water_box.atoms.write('water_box.pdb')

If you are not interested in the exact number of water molecules, you can also specify a
target density:

.. code-block:: python

    water_box_with_density = solvate.InsertDensity(empty_box, spce, target_density=0.033)
    water_box_with_density.atoms.write('water_box_with_density.pdb')

All `Insert` methods attempt to place the projectiles randomly within the target while avoiding overlaps. Depending on the number of insertions and the size of the target, this process may take some time or even fail, if too many attempts have to be made.

In most cases you should use the `SolvatePlanar` method for solvating Targets, as it is optimized for this use case.
For more detailed information on the available methods and options, please refer to the API documentation.