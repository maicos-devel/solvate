#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Build universes from template molecules."""

from typing import Optional

from matplotlib.path import Path

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.constants import N_A, epsilon_0, k as kB, e as e_el
from pint import UnitRegistry

ureg = UnitRegistry()
Q_ = ureg.Quantity

N_A       *= Q_('1/mol')
epsilon_0 *= Q_('F/m')
e_el      *= Q_('C')
kB        *= Q_('J/K')

from .models import empty


def tile_universe(
    universe: mda.Universe,
    n_x: int,
    n_y: int,
    n_z: int,
) -> mda.Universe:
    """Returns a new Universe with `n_x * n_y * n_z` copies of the input."""
    box = universe.dimensions[:3]
    copied = []
    i = 0
    for x in tqdm(range(n_x)):
        for y in range(n_y):
            for z in range(n_z):
                u_ = universe.copy()
                move_by = box * (x, y, z)
                u_.residues.resids += len(universe.residues) * i
                u_.atoms.translate(move_by)
                copied.append(u_.atoms)
                i += 1

    new_universe = mda.Merge(*copied)
    new_box = box * (n_x, n_y, n_z)
    new_universe.dimensions = list(new_box) + [90] * 3
    return new_universe


def pos_random(InsertionDomain: np.ndarray) -> np.ndarray:
    """Returns a random position within the given domain."""
    return np.array(
        np.random.rand(3) * (InsertionDomain[3:6] - InsertionDomain[0:3])
        + InsertionDomain[0:3],
        dtype=np.float32,
    )


def rot_random() -> tuple:
    """Returns a random rotation angle and vector, sampled uniformly on a sphere."""
    u_1, u_2, u_3 = np.random.rand(3)

    theta, phi = np.arccos(2 * u_1 - 1), 2 * np.pi * u_2

    rot_vec = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    rot_angle = 360 * u_3

    return rot_angle, rot_vec


def SolvateCylinder(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    density: Optional[float] = None,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    min: float = 0,
    max: Optional[float] = None,
    dim: int = 2,
    distance: float = 1.25,
    tries: int = 1000,
    fudge_factor: float = 1,
) -> mda.Universe:
    """Inserts `n` projectile atoms in a cylindrical zone (fast)."""
    print(f"The fudge factor is {fudge_factor}")
    if max is None:
        max = TargetUniverse.dimensions[dim]

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget[:3] / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()
    pos[dim] = min

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    if density is not None:
        n = np.floor(density * (2 * radius) ** 2 * (max - min))
        solvate_by_density_flag = True
    else:
        solvate_by_density_flag = False

    dimensionsTarget = TargetUniverse.dimensions.copy()

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

    InsertionDomain = np.array(
        [pos[0] - radius, pos[1] - radius, min, pos[0] + radius, pos[1] + radius, max],
        dtype=np.float32,
    )

    InsertionVolume = (max - min) * np.pi * radius**2
    density = n / InsertionVolume

    SolvatedUniverse = SolvatePlanar(
        TargetUniverse,
        ProjectileUniverse,
        0,
        density,
        xmin=InsertionDomain[0],
        ymin=InsertionDomain[1],
        zmin=InsertionDomain[2],
        xmax=InsertionDomain[3],
        ymax=InsertionDomain[4],
        zmax=InsertionDomain[5],
        distance=distance,
        tries=tries,
        fudge_factor=fudge_factor,
    )
    dims = SolvatedUniverse.dimensions
    TargetAtoms = SolvatedUniverse.atoms[:nAtomsTarget]
    ProjectileAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
    atomsInside = (
        np.linalg.norm((ProjectileAtoms.positions - pos)[:, :2], axis=1) < radius
    )
    if TargetAtoms.n_atoms == 0:
        SolvatedUniverse = ProjectileAtoms[atomsInside].residues.atoms
    else:
        SolvatedUniverse = mda.Merge(
            TargetAtoms, ProjectileAtoms[atomsInside].residues.atoms
        )
    SolvatedUniverse.dimensions = dims
    print("Resulting number of atoms:", SolvatedUniverse.atoms.n_atoms)
    print(
        "Resulting number of projectiles:",
        (SolvatedUniverse.atoms.n_atoms - nAtomsTarget) / nAtomsProjectile,
    )

    missingProjectiles = int(
        ((n * nAtomsProjectile + nAtomsTarget) - SolvatedUniverse.atoms.n_atoms)
        / nAtomsProjectile
    )
    print("Missing", missingProjectiles, "Projectiles.")
    if solvate_by_density_flag:
        print(f" {SolvatedUniverse.atoms.n_atoms - nAtomsTarget} projectiles inserted")
        return SolvatedUniverse
    if missingProjectiles > 0:
        print("Missing", missingProjectiles, "Projectiles.")
        print("Adjusting fudge factor and trying again.")
        new_fudge_factor = fudge_factor + 0.5
        return SolvateCylinder(
            TargetUniverse,
            ProjectileUniverse,
            n,
            density=None,
            pos=pos,
            radius=radius,
            min=min,
            max=max,
            dim=dim,
            distance=distance,
            tries=tries,
            fudge_factor=new_fudge_factor,
        )

    elif missingProjectiles < 0:
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        print("Too many projectiles inserted:", -missingProjectiles)
        print(nonTargetAtoms.n_atoms)
        print(nonTargetAtoms.residues.n_residues)
        print(np.unique(nonTargetAtoms.residues.resids).shape)
        print("Removing", -missingProjectiles, "randomly selected projectiles.")
        ToBeRemoved = nonTargetAtoms.residues[
            np.random.choice(
                np.arange(len(nonTargetAtoms.residues)),
                -missingProjectiles,
                replace=False,
            )
        ]
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - ToBeRemoved.atoms)
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        TargetAtoms = SolvatedUniverse.atoms[:nAtomsTarget]
        print(
            len(TargetAtoms.residues),
            len(nonTargetAtoms.residues),
            len(SolvatedUniverse.residues),
        )
        SolvatedUniverse.residues.resids = np.concatenate(
            [
                TargetAtoms.residues.resids,
                np.arange(
                    len(TargetAtoms.residues) + 1, len(SolvatedUniverse.residues) + 1
                ),
            ]
        )
        SolvatedUniverse.dimensions = dimensionsTarget
        print("Final number of atoms:", SolvatedUniverse.atoms.n_atoms)
        return SolvatedUniverse
    else:
        print("All projectiles inserted correctly")
        return SolvatedUniverse


def SolvatePlanar(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    density: Optional[float] = None,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    solvate_factor: int = 100,
    fudge_factor: float = 1.0,
    tries: int = 1000,
) -> mda.Universe:
    """Returns a rectacular box of `n` projectile atoms (fast).
    
    Solvates a universe by creating smaller universed (tiles) that are solvated
    individually and then tiled to create a big solvated box that is then inserted
    into the target universe. This method is much faster than inserting all solvent
    molecules one by one, especially for large numbers of solvent molecules. If the
    calculated number of tiles is only one, the function falls back to direct insertion.
    If too many projectiles are removed due to close contacts (distance keyword), the
    fudge factor is increased and the function is called recursively until at least n
    projectiles are inserted. Excess projectiles are removed afterwards to exactly
    match n.
    
    Keyword arguments:
    TargetUniverse     -- Universe to be solvated
    ProjectileUniverse -- Universe of the solvent molecule to be inserted
    n                  -- Number of solvent molecules to insert
    density            -- Density of solvent to insert (overrides n)
    xmin, ymin, zmin   -- Minimum coordinates of the insertion domain
    xmax, ymax, zmax   -- Maximum coordinates of the insertion domain
    distance           -- Minimum distance between inserted and target atoms
    solvate_factor     -- Initial guess for how many projectiles to insert per tile
    fudge_factor       -- Adjustment factor for the solvation factor
    tries              -- Number of attempts to insert each molecule

    Returns:
    mda.Universe       -- Solvated universe
    """
    # Use no fewer than 20 atoms for solvation
    SOLVATION_THRESHOLD = 20

    if xmax is None:
        xmax = TargetUniverse.dimensions[0]
    if ymax is None:
        ymax = TargetUniverse.dimensions[1]
    if zmax is None:
        zmax = TargetUniverse.dimensions[2]
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if zmin is None:
        zmin = 0

    InsertionDomain = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomainSize = InsertionDomain[3:6] - InsertionDomain[0:3]
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if density is not None:
        n = np.floor(
            density
            * InsertionDomainSize[0]
            * InsertionDomainSize[1]
            * InsertionDomainSize[2]
        )

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

    print(f"Should solvate {n} Projectiles")
    x = np.ceil((n / (solvate_factor * fudge_factor)) ** (1 / 3)).astype(int)

    # If only one tile is needed, use direct insertion
    if x <= 1:
        x = 1
        print(f"Solvation factor: {solvate_factor}")
        print(f"Best tiling is {x}x{x}x{x}.")

        return InsertPlanar(
            TargetUniverse,
            ProjectileUniverse,
            n,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            distance,
            tries,
        )
    # Make sure we have enough particles per tile
    # If not, reduce the number of tiles
    if n / (x**3) < SOLVATION_THRESHOLD and x > 2:
        x -= 1

    # Recalculate the real solvation factor
    real_solvate_factor = n / (x**3)

    print(f"Solvation factor: {solvate_factor}")
    print(f"Best tiling is {x}x{x}x{x}.")

    real_solvate_factor = np.ceil(real_solvate_factor * fudge_factor).astype(int)

    print("Real solvation factor is", real_solvate_factor)
    print(
        "This results in a total of",
        x**3 * (real_solvate_factor),
        "projectiles in the solvate box",
    )
    # Calculate the dimensions of the small solvate box
    solvate_box_dimensions = np.concatenate(
        [InsertionDomainSize / x, dimensionsTarget[3:6]]
    )

    # Create the small solvate box
    solvate_box = InsertPlanar(
        empty(solvate_box_dimensions),
        ProjectileUniverse,
        real_solvate_factor,
        distance=distance,
        tries=tries * 1000,
    )

    # We tile the small box to make a big box that is big enough to contain
    # the insertion domain
    print("Tiling solvate box...")
    big_solvate_box = tile_universe(solvate_box, x, x, x)

    # Shift the solvate box to the beginning of the insertion domain
    big_solvate_box.atoms.translate(InsertionDomain[0:3])

    print("Inserting solvate box into target universe...")

    nAtomsSolvate = big_solvate_box.atoms.n_atoms

    print("Target atoms:", nAtomsTarget)
    print("Projectile atoms:", nAtomsSolvate)

    if nAtomsTarget == 0:
        SolvatedUniverse = big_solvate_box
    else:
        SolvatedUniverse = mda.Merge(TargetUniverse.atoms, big_solvate_box.atoms)
    SolvatedUniverse.dimensions = dimensionsTarget
    target = SolvatedUniverse.atoms[0:nAtomsTarget]
    projectile = SolvatedUniverse.atoms[-nAtomsSolvate:]

    print("Search for overlapping atoms...")

    ns = mda.lib.NeighborSearch.AtomNeighborSearch(
        projectile, SolvatedUniverse.dimensions
    )
    touching_atoms = ns.search(target, distance, level="R").atoms
    if touching_atoms.n_atoms > 0:
        # touching_atoms = touching_atoms.intersection(projectile).residues.atoms
        # if touching_atoms.n_atoms / nAtomsProjectile:

        print(
            "Removing touching projectiles:", touching_atoms.n_atoms / nAtomsProjectile
        )
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - touching_atoms)
        SolvatedUniverse.dimensions = dimensionsTarget
    print("Resulting number of atoms:", SolvatedUniverse.atoms.n_atoms)
    print("Expected number of atoms:", n * nAtomsProjectile + nAtomsTarget)
    missingProjectiles = int(
        ((n * nAtomsProjectile + nAtomsTarget) - SolvatedUniverse.atoms.n_atoms)
        / nAtomsProjectile
    )

    if density is not None:
        print(f" {SolvatedUniverse.atoms.n_atoms - nAtomsTarget} projectiles inserted")
        return SolvatedUniverse
    if missingProjectiles > 0:
        print("Missing", missingProjectiles, "Projectiles.")
        print("Adjusting fudge factor and trying again.")
        return SolvatePlanar(
            TargetUniverse,
            ProjectileUniverse,
            n,
            density,
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            distance,
            solvate_factor,
            fudge_factor + 10 * missingProjectiles / n,
            tries,
        )
    elif missingProjectiles < 0:
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        print("Too many projectiles inserted:", -missingProjectiles)
        print(nonTargetAtoms.n_atoms)
        print(nonTargetAtoms.residues.n_residues)
        print(np.unique(nonTargetAtoms.residues.resids).shape)
        print("Removing", -missingProjectiles, "randomly selected projectiles.")
        ToBeRemoved = nonTargetAtoms.residues[
            np.random.choice(
                np.arange(len(nonTargetAtoms.residues)),
                -missingProjectiles,
                replace=False,
            )
        ]
        SolvatedUniverse = mda.Merge(SolvatedUniverse.atoms - ToBeRemoved.atoms)
        nonTargetAtoms = SolvatedUniverse.atoms[nAtomsTarget:]
        TargetAtoms = SolvatedUniverse.atoms[:nAtomsTarget]
        print(
            len(TargetAtoms.residues),
            len(nonTargetAtoms.residues),
            len(SolvatedUniverse.residues),
        )
        SolvatedUniverse.residues.resids = np.concatenate(
            [
                TargetAtoms.residues.resids,
                np.arange(
                    len(TargetAtoms.residues) + 1, len(SolvatedUniverse.residues) + 1
                ),
            ]
        )
        SolvatedUniverse.dimensions = dimensionsTarget
        print("Final number of atoms:", SolvatedUniverse.atoms.n_atoms)
        return SolvatedUniverse
    else:
        print("All projectiles inserted correctly")
        return SolvatedUniverse


def InsertPlanar(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    tries: int = 1000,
) -> mda.Universe:
    """Inserts `n` projectile atoms in a rectangular zone."""
    InsertionDomain = [xmin, ymin, zmin, xmax, ymax, zmax]
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomain = np.array(InsertionDomain)
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    if TargetUniverse.atoms.n_atoms == 0:
        TargetUniverse = ProjectileUniverse.copy()
        TargetUniverse.dimensions = dimensionsTarget
        TargetUniverse.atoms.translate(
            pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry()
        )
        TargetUniverse.atoms.rotateby(*rot_random())
        n -= 1

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms

        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]
        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target, dimensionsTarget)

        for _attempt in range(tries):
            projectile.translate(
                pos_random(InsertionDomain) - projectile.atoms.center_of_geometry()
            )

            projectile.rotateby(*rot_random())

            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found,\
                maybe you are trying to insert to many particles? Aborting."
            )

        projectile.residues.resids = (
            projectile.residues.resids + target.residues.resids[-1]
        )

    return TargetUniverse


def InsertCylinder(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    min: float = 0,
    max: Optional[float] = None,
    dim: int = 2,
    distance: float = 1.25,
    tries: int = 1000,
) -> mda.Universe:
    """Inserts `n` projectile atoms in a cylindrical zone."""
    if max is None:
        max = TargetUniverse.dimensions[dim]

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()
    pos[dim] = min

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for _attempt in range(tries):
            projectile.rotateby(*rot_random())

            r = radius * np.sqrt(np.random.rand())
            phi, z = np.random.rand(2) * [2 * np.pi, (max - min)]
            newcoord = np.roll([r * np.cos(phi), r * np.sin(phi), z], dim - 2) + pos

            projectile.translate(newcoord - projectile.atoms.center_of_geometry())

            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found,\
                maybe you are trying to insert too many particles? Aborting."
            )

        projectile.residues.resids = (
            projectile.residues.resids + target.residues.resids[-1]
        )

    return TargetUniverse


def InsertSphere(
    TargetUniverse: mda.Universe,
    ProjectileUniverse: mda.Universe,
    n: int = 1,
    pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    tries: int = 1000,
) -> mda.Universe:
    """Inserts `n` projectile atoms in a spherical zone."""

    def rand_spherical(radius: float = 1.0) -> np.ndarray:
        u = np.random.rand()
        v = np.random.rand()

        theta = u * 2.0 * np.pi
        phi = np.arccos(2.0 * v - 1.0)
        r = radius * np.power(np.random.rand(), 1 / 3)

        sinTheta = np.sin(theta)
        cosTheta = np.cos(theta)
        sinPhi = np.sin(phi)
        cosPhi = np.cos(phi)

        x = r * sinPhi * cosTheta
        y = r * sinPhi * sinTheta
        z = r * cosPhi
        return np.array([x, y, z])

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos is None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget[:3] / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()

    if radius is None:
        radius = np.min(dimensionsTarget) / 2

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    if TargetUniverse.atoms.n_atoms == 0:
        TargetUniverse = ProjectileUniverse.copy()
        TargetUniverse.dimensions = dimensionsTarget
        TargetUniverse.atoms.translate(
            pos + rand_spherical(radius) - TargetUniverse.atoms.center_of_geometry()
        )
        TargetUniverse.atoms.rotateby(*rot_random())
        n -= 1

    for _N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for _attempt in range(tries):
            projectile.rotateby(*rot_random())
            newcoord = rand_spherical(radius) + pos
            projectile.translate(newcoord - projectile.atoms.center_of_geometry())
            if len(ns.search(projectile, distance)) == 0:
                break
        else:
            raise RuntimeError(
                "Error: No suitable position found, \
                maybe you are trying to insert to many particles? Aborting."
            )
        projectile.residues.resids = (
            projectile.residues.resids + target.residues.resids[-1]
        )

    return TargetUniverse


def SolvatePlanarPB(
    TargetUniverse: mda.Universe,
    AnionProjectileUniverse: mda.Universe,
    CationProjectileUniverse: mda.Universe,
    N_anions: int,
    N_cations: int,
    epsilon_r: float = 80.2,
    T: float = 300.0,
    q_diff: int = 0,
    xmin: int = 0,
    ymin: int = 0,
    zmin: int = 0,
    xmax: Optional[float] = None,
    ymax: Optional[float] = None,
    zmax: Optional[float] = None,
    distance: float = 1.25,
    fudge_factor: float = 1.5,
    tries: int = 100,
    plot: bool = False,
    output_path: Optional[str] = None,
) -> mda.Universe:
    """
    Inserts ions into a system with plate capacitor geometry according to a Poisson-Boltzmann distribution.

    Positional arguments:
    TargetUniverse           -- MDAnalysis Universe of the target system.
    AnionProjectileUniverse  -- MDAnalysis Universe of the anion projectile.
    CationProjectileUniverse -- MDAnalysis Universe of the cation projectile.
    N_anions                 -- Number of anions to insert.
    N_cations                -- Number of cations to insert.

    Keyword arguments:
    epsilon_r                -- Relative permittivity of the medium.
    T                        -- Temperature in Kelvin.
    q_diff                   -- Total charge difference between cations and anions.
    xmin, ymin, zmin         -- Minimum coordinates of the insertion domain.
    xmax, ymax, zmax         -- Maximum coordinates of the insertion domain.
    distance                 -- Minimum distance between inserted ions and existing atoms.
    fudge_factor             -- Fudge factor for number of inserted ions.
    tries                    -- Number of attempts to find a valid insertion position.

    Returns:
    Solvated Universe with inserted ions.
    """

    def lambda_D(epsilon_r, T, cN_bulk_cat, cN_bulk_an):
        """
        Calculate the Debye length.
        
        Positional arguments:
        epsilon_r   -- Relative permittivity of the medium.
        T           -- Temperature.
        cN_bulk_cat -- Bulk concentration of cations.
        cN_bulk_an  -- Bulk concentration of anions.
        
        Returns:
        Debye length.
        """

        return np.sqrt(epsilon_r * epsilon_0 * kB * T / (2 * e_el**2 * (cN_bulk_cat + cN_bulk_an)/2))

    def phi_z(sigma, lambda_D, z, epsilon_r):
        """
        Calculate the electrostatic potential profile in z-direction.
        
        Positional arguments:
        sigma      -- Surface charge density.
        lambda_D   -- Debye length.
        z          -- Distance from the charged surface.
        epsilon_r  -- Relative permittivity of the medium.
        
        Returns:
        Electrostatic potential profile in z-direction.
        """

        return sigma / (epsilon_r * epsilon_0) * lambda_D * np.exp(-z / lambda_D)

    def pb_factor_profile(q, T, phi):
        """
        Calculate the Poisson-Boltzmann factor profile in z-direction.
        
        Positional arguments:
        q    -- Charge of the ion.
        T    -- Temperature.
        phi  -- Electrostatic potential profile.
        
        Returns:
        Poisson-Boltzmann factor profile in z-direction.
        """

        p = np.exp(-q * phi / (kB * T))
        p /= np.sum(p)
        return p

    def generate_z_positions(pbfp, N, z):
        """
        Generate z positions based on the Poisson-Boltzmann factor.
        
        Positional arguments:
        pbfp -- Poisson-Boltzmann factor profile.
        N  -- Number of ions to place.
        z  -- z positions corresponding to the Boltzmann factor profile.
        
        Returns:
        z positions of the ions.
        """

        pbfp = pbfp.to('dimensionless').magnitude
        samples = np.random.choice(len(pbfp), size=N, p=pbfp)

        # Convert indices to z positions
        z_positions = z[samples]
        return z_positions

    def insert_ions(TargetUniverse, ProjectileUniverse, z_positions, distance, tries):
        """
        Insert ions into the target universe at specified z positions.

        Positional arguments:
        TargetUniverse   -- The universe to insert ions into.
        ProjectileUniverse -- The universe containing the ions to insert.
        z_positions       -- The z positions to insert the ions at.
        distance          -- Minimum distance between inserted ions and existing atoms.
        tries             -- Number of attempts to find a valid insertion position.

        Returns:
        Updated TargetUniverse with inserted ions.
        """

        nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

        # No pint units for MDAnalysis
        distance = distance.to('angstrom').magnitude
        z_positions = z_positions.to('angstrom').magnitude

        if TargetUniverse.atoms.n_atoms == 0:
            TargetUniverse = ProjectileUniverse.copy()
            TargetUniverse.dimensions = dimensionsTarget

            t_vec = pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry()
            first_z_position = z_positions[0]
            z_positions = np.delete(z_positions, 0)


            t_vec[2] = first_z_position - ProjectileUniverse.atoms.center_of_geometry()[2]
            TargetUniverse.atoms.translate(
                t_vec
            )
            TargetUniverse.atoms.rotateby(*rot_random())

        for _N, z in tqdm(enumerate(z_positions)):
            nAtomsTarget = TargetUniverse.atoms.n_atoms

            TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
            TargetUniverse.dimensions = dimensionsTarget

            target = TargetUniverse.atoms[0:nAtomsTarget]
            projectile = TargetUniverse.atoms[-nAtomsProjectile:]
            ns = mda.lib.NeighborSearch.AtomNeighborSearch(target, dimensionsTarget)

            for _attempt in range(tries):
                t_vec = pos_random(InsertionDomain) - projectile.atoms.center_of_geometry()
                t_vec[2] = z - projectile.atoms.center_of_geometry()[2]
                projectile.translate(
                    t_vec
                )

                projectile.rotateby(*rot_random())

                if len(ns.search(projectile, distance)) == 0:
                    break
            else:
                raise RuntimeError(
                    "Error: No suitable position found,\
                    maybe you are trying to insert to many particles? Aborting."
                )

            projectile.residues.resids = (
                projectile.residues.resids + target.residues.resids[-1]
            )
        return TargetUniverse

    if output_path is not None:
        from pathlib import Path
        try:
            p = Path(output_path)
        except Exception as e:
            print(f"Error processing output_path: {e}")
    else:
        p = None
    # Use pint units
    T = T * ureg('K')
    q_diff = q_diff * ureg('elementary_charge')
    distance = distance * ureg('angstrom')

    q_excess = (N_cations - N_anions) * ureg('elementary_charge')
    print(f"Total excess charge to be compensated: {q_excess.magnitude} e")
    n_ion    = np.argmin([N_cations, N_anions])

    # Calculate plate charges
    if q_excess != 0:
        q_1 = (-q_diff + q_excess) / 2 
        q_2 = (q_diff + q_excess) / 2
    else:
        q_1 = -(q_diff / 2)
        q_2 = (q_diff / 2)

    # Check TargetUniverse dimensions
    if TargetUniverse.dimensions is None:
        raise ValueError("TargetUniverse must have defined dimensions.")
    if xmax is None:
        xmax = TargetUniverse.dimensions[0]
    if ymax is None:
        ymax = TargetUniverse.dimensions[1]
    if zmax is None:
        zmax = TargetUniverse.dimensions[2]
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if zmin is None:
        zmin = 0

    # Define insertion domain
    InsertionDomain = np.array([xmin, ymin, zmin, xmax, ymax, zmax])
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomainSize = (InsertionDomain[3:6] - InsertionDomain[0:3]) * Q_('angstrom')
    dimensionsTarget = TargetUniverse.dimensions.copy()

    # Calculate surface charge densities
    sigma_1 = q_1 / (InsertionDomainSize[0] * InsertionDomainSize[1])
    sigma_2 = q_2 / (InsertionDomainSize[0] * InsertionDomainSize[1])
    print(f"Surface charge density 1: {sigma_1.to('elementary_charge / angstrom^2').magnitude:.5f} e/Å²")
    print(f"Surface charge density 2: {sigma_2.to('elementary_charge / angstrom^2').magnitude:.5f} e/Å²")

    # Calculate bulk concentration --> needed for Debye length
    volume = (InsertionDomainSize[0]
              * InsertionDomainSize[1]
              * InsertionDomainSize[2])
    cN_bulk_cat = N_cations / volume
    cN_bulk_an  = N_anions / volume

    # Create a grid of z values
    z = np.linspace(0, InsertionDomainSize[2], 500)

    # Calculate Debye length
    l_D = lambda_D(epsilon_r, T, cN_bulk_cat, cN_bulk_an)
    print(f"Debye length : {l_D.to('angstrom').magnitude:.2f} Å")

    # Calculate potential profiles
    phi_1 = phi_z(sigma_1, l_D, z, epsilon_r)
    phi_2 = phi_z(sigma_2, l_D, InsertionDomainSize[2] - z, epsilon_r)
    phi_total = phi_1 + phi_2
    print(f'Potential difference between plates: { (phi_total[0] - phi_total[-1]).to("volt").magnitude:.2f} V')

    # Calculate Poisson-Boltzmann factor profiles
    pbfp_anions = pb_factor_profile(e_el, T, phi_total)
    pbfp_cations = pb_factor_profile(-e_el, T, phi_total)

    if plot:
        import matplotlib.pyplot as plt
        # Plotting
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6))
        axes[0].axvline(x=0, color='k', linestyle='-')
        axes[0].axvline(x=InsertionDomainSize[2].magnitude, color='k', linestyle='-')
        axes[0].axvline(x=l_D.to('angstrom').magnitude, color='grey', linestyle='--')
        axes[0].set_xlim(-5, InsertionDomainSize[2].magnitude + 5)
        axes[0].plot(z.magnitude, phi_total.to('volt').magnitude, label='Total Potential', color='green', linestyle='-')
        axes[0].set_xlabel('z insertion domain / Å', fontsize=14)
        axes[0].set_ylabel(r'Electrostatic Potential $\Phi$ / V', fontsize=14)
        axes[0].grid()

        axes[1].plot(z.magnitude, pbfp_anions.magnitude, label='Anions', linestyle='None', marker='o', markersize=2, color='red')
        axes[1].plot(z.magnitude, pbfp_cations.magnitude, label='Cations', linestyle='None', marker='o', markersize=2, color='blue')
        axes[1].set_ylabel('Probability / a.u.', fontsize=14)
        axes[1].set_xlabel('z insertion domain / Å', fontsize=14)
        axes[1].legend(fontsize=14)
        axes[1].axvline(x=0, color='grey', linestyle='--')
        axes[1].axvline(x=InsertionDomainSize[2].magnitude, color='grey', linestyle='--')
        axes[1].grid()
        axes[1].set_xlim(-5, InsertionDomainSize[2].magnitude + 5)

        if p:
            fig.savefig(p / f'an_{N_anions}-cat_{N_cations}-qdiff_{q_diff.magnitude}-Phi_P.png', dpi=600)

        fig.show()

    # Draw more than needed ions to account for rejections during insertion
    N_anions_to_draw = np.ceil(N_anions * fudge_factor).astype(int)
    N_cations_to_draw = np.ceil(N_cations * fudge_factor).astype(int)

    # Generate z positions based on Poisson-Boltzmann distribution
    z_positions_anions = generate_z_positions(pbfp_anions, N_anions_to_draw, z)
    z_positions_cations = generate_z_positions(pbfp_cations, N_cations_to_draw, z)

    # Remove positions that are too close to the plates
    z_positions_anions = z_positions_anions[
        (z_positions_anions > distance) &
        (z_positions_anions < (InsertionDomainSize[2] - distance))
    ]
    z_positions_cations = z_positions_cations[
        (z_positions_cations > distance) &
        (z_positions_cations < (InsertionDomainSize[2] - distance))
    ]

    # Select only the required number of ions
    z_positions_anions = z_positions_anions[0:N_anions]
    z_positions_cations = z_positions_cations[0:N_cations]

    print(len(z_positions_anions), "anions to be inserted.")
    print(len(z_positions_cations), "cations to be inserted.")

    # Adjust z positions to absolute coordinates (to be improved)
    z_positions_anions = z_positions_anions + zmin * ureg('angstrom')
    z_positions_cations = z_positions_cations + zmin * ureg('angstrom')

    # Insert ions into the TargetUniverse
    TargetUniverse = insert_ions(TargetUniverse, AnionProjectileUniverse, z_positions_anions, distance, tries)
    TargetUniverse = insert_ions(TargetUniverse, CationProjectileUniverse, z_positions_cations, distance, tries)

    # Plot final z position distributions
    max_counts = max(
        np.histogram(z_positions_anions.to('angstrom').magnitude, bins=50)[0].max(),
        np.histogram(z_positions_cations.to('angstrom').magnitude, bins=50)[0].max()
    )
    if plot:
        fig, axes = plt.subplots(ncols=2, figsize=(16, 6))

        axes[0].hist(z_positions_anions.to('angstrom').magnitude, bins=50, color='red')
        axes[0].set_xlabel('z / Å', fontsize=14)
        axes[0].set_ylabel('Count / a.u.', fontsize=14)
        axes[0].legend(['Anions'], fontsize=14)
        axes[0].axvline(x=zmin, color='grey', linestyle='--')
        axes[0].axvline(x=zmax, color='grey', linestyle='--')
        axes[0].grid()
        axes[0].set_xlim(0, dimensionsTarget[2])
        axes[0].set_ylim(0, max_counts + 1)

        axes[1].hist(z_positions_cations.to('angstrom').magnitude, bins=50, color='blue')
        axes[1].set_xlabel('z / Å', fontsize=14)
        axes[1].set_ylabel('Count / a.u.', fontsize=14)
        axes[1].legend(['Cations'], fontsize=14)
        axes[1].axvline(x=zmin, color='grey', linestyle='--')
        axes[1].axvline(x=zmax, color='grey', linestyle='--')
        axes[1].grid()
        axes[1].set_xlim(0, dimensionsTarget[2])
        axes[1].set_ylim(0, max_counts + 1)

        if p:
            fig.savefig(p / f'an_{N_anions}-cat_{N_cations}-qdiff_{q_diff.magnitude}-final_distribution.png', dpi=600)

        plt.show()

    return TargetUniverse