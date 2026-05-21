#!/usr/bin/env python
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Predefined molecules."""

import warnings

import MDAnalysis as mda
import numpy as np


def _three_site_molecule(theta: float) -> tuple:
    """Returns three coordinates for use in a three-site water model.

    Parameters
    ----------
    theta (float): The angle in radians for the third coordinate. Range (0, pi].

    Returns
    -------
    tuple: A tuple containing three numpy arrays representing the coordinates of the
    three sites.
           - pos_O: The origin coordinate [0, 0, 0].
           - pos_H1: The second coordinate [1, 0, 0].
           - pos_H2: The third coordinate at an angle theta in the xy plane.

    Raises
    ------
    ValueError: If theta is not in the range (0, pi].
    """
    if theta <= 0 or theta > np.pi:
        raise ValueError("theta must be in (0, pi]")
    pos_O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    pos_H1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    pos_H2 = np.array([np.cos(theta), -np.sin(theta), 0.0], dtype=np.float32)

    return pos_O, pos_H1, pos_H2


def empty(dimensions: np.ndarray) -> mda.Universe:
    """Create an empty :class:`~MDAnalysis.core.universe.Universe`.

    Useful as a target for the ``Insert*`` and ``Solvate*`` functions when
    building a system from scratch.

    Parameters
    ----------
    dimensions : array_like of shape (6,)
        Simulation cell as ``[a, b, c, alpha, beta, gamma]`` with lengths in
        Å and angles in degrees.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        An empty universe with no atoms and the given simulation cell.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="""Residues specified but no atom_resindex given. """
            """All atoms will be placed in first Residue.""",
        )
        warnings.filterwarnings(
            "ignore",
            message="""Segments specified but no segment_resindex given. """
            """All residues will be placed in first Segment""",
        )
        u = mda.Universe.empty(0, trajectory=True)

    u.dimensions = dimensions
    return u


def type_a(
    l_1: float,
    q_O: float,
    q_H: float,
    theta: float,
    mass_O: float = 15.999,
    mass_H: float = 1.00784,
) -> mda.Universe:
    """Build a single 3-site water molecule (type *a*).

    The oxygen sits at the origin; the two hydrogens lie in the xy plane at
    distance ``l_1`` from the oxygen and at an angle ``theta`` to each
    other. Topology, masses, and partial charges are populated so the
    resulting universe is ready to be replicated by an ``Insert*`` or
    ``Solvate*`` call.

    Parameters
    ----------
    l_1 : float
        O–H bond length, in Å.
    q_O : float
        Partial charge on the oxygen, in units of the elementary charge.
    q_H : float
        Partial charge on each hydrogen, in units of the elementary charge.
    theta : float
        H–O–H angle, in radians. Must lie in ``(0, pi]``.
    mass_O : float, default 15.999
        Oxygen mass, in u.
    mass_H : float, default 1.00784
        Hydrogen mass, in u.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        A single-residue universe with three atoms (``OW``, ``HW1``,
        ``HW2``) and the appropriate bond and angle topology.

    See Also
    --------
    spce, tip3p : Pre-parameterised 3-site water models.
    type_c : 4-site variant with an additional M site.
    """
    model = mda.Universe.empty(
        3, n_residues=1, atom_resindex=[0, 0, 0], residue_segindex=[0], trajectory=True
    )

    model.add_TopologyAttr("name", ["OW", "HW1", "HW2"])
    model.add_TopologyAttr("type", ["O", "H", "H"])
    model.add_TopologyAttr("resname", ["SOL"])
    model.add_TopologyAttr("resid", [1])
    model.add_TopologyAttr("segid", ["SOL"])
    model.add_TopologyAttr("charges", [q_O, q_H, q_H])
    model.add_TopologyAttr("masses", [mass_O, mass_H, mass_H])
    model.add_TopologyAttr("bonds", [(0, 1), (0, 2)])
    model.add_TopologyAttr("angles", [(1, 0, 2)])

    pos_O, pos_H1, pos_H2 = _three_site_molecule(theta)

    pos_H1 *= l_1
    pos_H2 *= l_1

    model.atoms.positions = np.array([pos_O, pos_H1, pos_H2])

    return model


def type_c(
    l_1: float,
    l_2: float,
    q_M: float,
    q_H: float,
    theta: float,
    mass_O: float = 15.999,
    mass_H: float = 1.00784,
) -> mda.Universe:
    """Build a single 4-site water molecule (type *c*).

    Like :func:`type_a` but with an additional massless M site placed on the
    H–O–H angle bisector at distance ``l_2`` from the oxygen. The negative
    charge sits on the M site rather than the oxygen.

    Parameters
    ----------
    l_1 : float
        O–H bond length, in Å.
    l_2 : float
        O–M distance along the H–O–H bisector, in Å.
    q_M : float
        Partial charge on the M site, in units of the elementary charge.
    q_H : float
        Partial charge on each hydrogen, in units of the elementary charge.
    theta : float
        H–O–H angle, in radians. Must lie in ``(0, pi]``.
    mass_O : float, default 15.999
        Oxygen mass, in u.
    mass_H : float, default 1.00784
        Hydrogen mass, in u.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        A single-residue universe with four atoms (``OW``, ``HW1``, ``HW2``,
        ``MW``), bonds, and angle topology.

    See Also
    --------
    tip4p_epsilon : Pre-parameterised 4-site water model.
    type_a : 3-site variant.
    """
    model = mda.Universe.empty(
        4,
        n_residues=1,
        atom_resindex=[0, 0, 0, 0],
        residue_segindex=[0],
        trajectory=True,
    )

    model.add_TopologyAttr("name", ["OW", "HW1", "HW2", "MW"])
    model.add_TopologyAttr("type", ["O", "H", "H", "D"])
    model.add_TopologyAttr("resname", ["SOL"])
    model.add_TopologyAttr("resid", [1])
    model.add_TopologyAttr("segid", ["SOL"])
    model.add_TopologyAttr("charges", [0, q_H, q_H, q_M])
    model.add_TopologyAttr("masses", [mass_O, mass_H, mass_H, 0])
    model.add_TopologyAttr("bonds", [(0, 1), (0, 2)])
    model.add_TopologyAttr("angles", [(1, 0, 2)])

    pos_O, pos_H1, pos_H2 = _three_site_molecule(theta)

    angle_bisector = (pos_H1 + pos_H2) / np.linalg.norm(pos_H1 + pos_H2)

    pos_H1 *= l_1
    pos_H2 *= l_1
    pos_M = l_2 * angle_bisector

    model.atoms.positions = np.array([pos_O, pos_H1, pos_H2, pos_M])

    return model


def spce() -> mda.Universe:
    """Return a single SPC/E water molecule.

    Uses an O–H bond length of 1.0 Å, an H–O–H angle of 109.47°, and the
    hydrogen partial charge ``q_H = 0.4238 e``.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        Single-residue universe with three atoms representing SPC/E water.

    See Also
    --------
    tip3p, tip4p_epsilon
    """
    l_1 = 1

    q_H = 0.4238
    q_O = -2 * q_H
    theta = np.deg2rad(109.47)

    return type_a(l_1, q_O, q_H, theta)


def tip4p_epsilon() -> mda.Universe:
    """Return a single TIP4P/ε water molecule.

    Uses an O–H bond length of 0.9572 Å, an O–M distance of 0.105 Å, an
    H–O–H angle of 104.52°, and a hydrogen partial charge of
    ``q_H = 0.527 e`` (the negative charge sits on the M site).

    Returns
    -------
    MDAnalysis.core.universe.Universe
        Single-residue universe with four atoms representing TIP4P/ε water.

    See Also
    --------
    spce, tip3p
    """
    l_1 = 0.9572
    l_2 = 0.105

    q_H = 0.5270
    q_M = -2 * q_H

    theta = np.deg2rad(104.52)

    return type_c(l_1, l_2, q_M, q_H, theta)


def tip3p():
    """Return a single TIP3P water molecule.

    Uses an O–H bond length of 0.9572 Å, an H–O–H angle of 104.52°, and a
    hydrogen partial charge of ``q_H = 0.417 e``.

    Returns
    -------
    MDAnalysis.core.universe.Universe
        Single-residue universe with three atoms representing TIP3P water.

    See Also
    --------
    spce, tip4p_epsilon
    """
    l_1 = 0.9572

    q_H = 0.417
    q_O = -2 * q_H
    theta = np.deg2rad(104.52)

    return type_a(l_1, q_O, q_H, theta)
