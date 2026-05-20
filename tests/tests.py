#!/usr/bin/env python
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the solvate package."""

import warnings

import MDAnalysis as mda
import numpy as np
import pytest
from numpy.testing import assert_allclose

import solvate
from solvate.insert import _renumber_projectile_resids


def _make_universe(atoms_per_res, resids):
    """Build a minimal universe with given residue layout and resids."""
    n_atoms = sum(atoms_per_res)
    n_res = len(atoms_per_res)
    atom_resindex = [i for i, n in enumerate(atoms_per_res) for _ in range(n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = mda.Universe.empty(
            n_atoms,
            n_residues=n_res,
            atom_resindex=atom_resindex,
            residue_segindex=[0] * n_res,
            trajectory=True,
        )
    u.add_TopologyAttr("resid", resids)
    u.atoms.positions = np.zeros((n_atoms, 3))
    return u


class TestInserts:
    """Test the insertion code."""

    @pytest.mark.parametrize("n_water", [1, 10, 100])
    def test_insert_planar_n_water(self, n_water):
        """Test the number of inserted particles in InsertPlanar."""
        emptyUniverse = solvate.models.empty([20, 20, 20, 90, 90, 90])
        testParticle = solvate.models.spce()
        u = solvate.InsertSphere(emptyUniverse, testParticle, n_water)
        assert u.atoms.n_atoms == n_water * 3

    # TODO(@hejamu): def test_insert_planar_density(self):
    #     """Test the density of the inserted particles in InsertPlanar."""

    # TODO(@hejamu): def test_insert_sphere_n_water(self):
    #     """Test the number of inserted particles in InsertSphere."""

    # TODO(@hejamu): def test_insert_sphere_density(self):
    #     """Test the density of the inserted particles in InsertSphere."""

    # TODO(@hejamu): def test_insert_cylinder_n_water(self):
    #     """Test the number of inserted particles in InsertCylinder."""

    # TODO(@hejamu): def test_insert_cylinder_density(self):
    #     """Test the density of the inserted particles in InsertCylinder."""

    # TODO(@hejamu): def test_insert_planar_domain(self):
    #     """Test the domain of the inserted particles in InsertPlanar."""

    # TODO(@hejamu): def test_insert_sphere_domain(self):
    #     """Test the domain of the inserted particles in InsertSphere."""

    # TODO(@hejamu): def test_insert_cylinder_domain(self):
    #     """Test the domain of the inserted particles in InsertCylinder."""

class TestInsertPlanar:
    """Tests for InsertPlanar."""

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_n_atoms(self, n):
        """InsertPlanar inserts exactly n * atoms_per_molecule atoms."""
        u = solvate.InsertPlanar(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            n,
        )
        assert u.atoms.n_atoms == n * 3

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_resids_empty_target(self, n):
        """Residues are numbered 1..n when inserted into an empty box."""
        u = solvate.InsertPlanar(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            n,
        )
        assert list(u.residues.resids) == list(range(1, n + 1))

    def test_resids_continue_from_target(self):
        """Projectile resids continue monotonically after target resids."""
        target = solvate.InsertPlanar(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            2,
        )
        u = solvate.InsertPlanar(target, solvate.models.spce(), 3)
        assert list(u.residues.resids) == [1, 2, 3, 4, 5]


class TestInsertSphere:
    """Tests for InsertSphere."""

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_n_atoms(self, n):
        """InsertSphere inserts exactly n * atoms_per_molecule atoms."""
        u = solvate.InsertSphere(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            n,
        )
        assert u.atoms.n_atoms == n * 3

    @pytest.mark.parametrize("n", [1, 5, 10])
    def test_resids_empty_target(self, n):
        """Residues are numbered 1..n when inserted into an empty box."""
        u = solvate.InsertSphere(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            n,
        )
        assert list(u.residues.resids) == list(range(1, n + 1))

    def test_resids_continue_from_target(self):
        """Projectile resids continue monotonically after target resids."""
        target = solvate.InsertSphere(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            2,
        )
        u = solvate.InsertSphere(target, solvate.models.spce(), 3)
        assert list(u.residues.resids) == [1, 2, 3, 4, 5]


class TestInsertCylinder:
    """Tests for InsertCylinder.

    InsertCylinder requires a non-empty target because it uses
    target.residues.resids[-1] on the first iteration.
    """

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_n_atoms(self, n):
        """InsertCylinder inserts exactly n molecules into a non-empty target."""
        target = solvate.InsertPlanar(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            1,
        )
        u = solvate.InsertCylinder(target, solvate.models.spce(), n)
        assert u.atoms.n_atoms == (1 + n) * 3

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_resids_contiguous(self, n):
        """Residues are numbered contiguously and monotonically after insertion."""
        target = solvate.InsertPlanar(
            solvate.models.empty([30, 30, 30, 90, 90, 90]),
            solvate.models.spce(),
            2,
        )
        u = solvate.InsertCylinder(target, solvate.models.spce(), n)
        assert list(u.residues.resids) == list(range(1, 2 + n + 1))


# class TestSolvate(object):
#     """Test the solvation code."""

# TODO(@hejamu): test_solvate_planar_n_water(self):
#     """Test the solvation of a planar system."""

# TODO(@hejamu): test_solvate_sphere_n_water(self):
#     """Test the solvation of a spherical system."""

# TODO(@hejamu): test_solvate_cylinder_n_water(self):
#     """Test the solvation of a cylindrical system."""

# TODO(@hejamu): test_solvate_planar_density(self):
#     """Test the density of the solvated system."""

# TODO(@hejamu): test_solvate_sphere_density(self):
#     """Test the density of the solvated system."""

# TODO(@hejamu): test_solvate_cylinder_density(self):
#     """Test the density of the solvated system."""


class TestModels:
    """Tests for the models."""

    # @pytest.fixture()
    # def ag(self):
    #     """Import MDA universe."""
    #     u = mda.Universe(WATER_TPR_NPT, WATER_TRR_NPT)
    #     return u.atoms

    # @pytest.mark.parametrize(
    #     "dens_type, mean", (("mass", 0.555), ("number", 0.093), ("charge", 2e-4))
    # )
    # def test_dens(self, ag, dens_type, mean):
    #     """Test density."""
    #     dens = DensitySphere(ag, dens=dens_type).run()
    #     assert_allclose(dens.results.profile.mean(), mean, atol=1e-4, rtol=1e-2)

    def test_type_a_general(self):
        """Test type a.

        Type a water molecules should have 3 atoms, 2 bonds, and 1 angle.
        """
        u = solvate.models.type_a(1, -2, 0.4238, np.deg2rad(109.47))
        assert u.atoms.n_atoms == 3
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1

    @pytest.mark.parametrize("angle", [45, 90, 125, 180])
    def test_three_site_angle(self, angle):
        """Test three site model build function."""
        pos_O, pos_H1, pos_H2 = solvate.models._three_site_molecule(np.deg2rad(angle))

        # We expect the middle atom to be at the origin for now.
        assert_allclose(pos_O, np.array([0.0, 0.0, 0.0]))
        # Make sure the vectors are normalized (ignoring floating point errors)
        assert_allclose(np.linalg.norm(pos_H1), 1.0)
        assert_allclose(np.linalg.norm(pos_H2), 1.0)
        # Calculate the angle
        alpha = np.arccos(np.dot(pos_H1, pos_H2))
        # If the angle is correct, we are happy
        assert_allclose(np.rad2deg(alpha), angle)

    @pytest.mark.parametrize("angle", [0, 181, -1])
    def test_three_site_error(self, angle):
        """Test three site model ValueError."""
        with pytest.raises(ValueError, match="theta must be in \\(0, pi\\]"):
            solvate.models._three_site_molecule(np.deg2rad(angle))

    @pytest.mark.parametrize("angle", [45, 90, 125, 180])
    def test_type_a_angle(self, angle):
        """Test type a angle."""
        u = solvate.models.type_a(1, -2, 1, np.deg2rad(angle))
        assert_allclose(u.atoms.angles[0].value(), angle)

    def test_type_c_general(self):
        """Test type c."""
        """Type c water molecules should have 4 atoms, 2 bonds, and 1 angle."""
        u = solvate.models.type_c(0.9572, 0.105, -1.054, 0.527, np.deg2rad(104.52))
        assert u.atoms.n_atoms == 4
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1

    @pytest.mark.parametrize("angle", [45, 90, 125, 180])
    def test_type_c_angle(self, angle):
        """Test type c angle."""
        u = solvate.models.type_c(0.9572, 0.105, -1.054, 0.527, np.deg2rad(angle))
        assert_allclose(u.atoms.angles[0].value(), angle)

    def test_spce(self):
        """Test SPC/E water.

        This is a regression test, the values here should NEVER change.
        """
        # TODO(@hejamu): Check the values against a source, this is a regression test
        u = solvate.models.spce()
        assert u.atoms.n_atoms == 3
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1
        assert_allclose(u.atoms.angles[0].value(), 109.47)
        assert_allclose(u.atoms.atoms.charges, [-0.8476, 0.4238, 0.4238])
        assert_allclose(u.atoms.atoms.masses, [15.999, 1.00784, 1.00784])

        # Check the positions, this depends on the orientation of the molecule,
        # so be extra careful when changing the default orientation of the models.
        ref_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0, 0.0],
                [-0.33331326, -0.94281614, 0.0],
            ],
            dtype=np.float32,
        )
        assert_allclose(u.atoms.atoms.positions, ref_pos)

    def test_tip4p_epsilon(self):
        """Test TIP4P/ε water.

        This is a regression test, the values here should NEVER change.
        """
        u = solvate.models.tip4p_epsilon()
        assert u.atoms.n_atoms == 4
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1
        assert_allclose(u.atoms.angles[0].value(), 104.52)
        assert_allclose(u.atoms.atoms.charges, [0, 0.527, 0.527, -1.054])
        assert_allclose(u.atoms.atoms.masses, [15.999, 1.00784, 1.00784, 0])

        # Check the positions, this depends on the orientation of the molecule,
        # so be extra careful when changing the default orientation of the models.
        ref_pos = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.23998721, -0.9266272, 0.0],
                [0.06426832, -0.08303362, 0.0],
            ],
            dtype=np.float32,
        )
        assert_allclose(u.atoms.atoms.positions, ref_pos)




class TestRenumberProjectileResids:
    """Tests for _renumber_projectile_resids."""

    def test_empty_target_starts_at_one(self):
        """With no target atoms, projectile resids start at 1."""
        u = _make_universe([3, 3], [7, 7])
        _renumber_projectile_resids(u, 0)
        assert list(u.residues.resids) == [1, 2]

    def test_projectile_starts_after_target_max(self):
        """Projectile resids start at max(target resids) + 1."""
        u = _make_universe([3, 3, 3], [5, 1, 1])
        _renumber_projectile_resids(u, 3)
        assert list(u.residues.resids) == [5, 6, 7]

    def test_out_of_order_target_uses_max(self):
        """Uses max() of target resids, not the last resid."""
        # Target resids are [5, 2, 3] — max is 5, so projectile starts at 6.
        u = _make_universe([1, 1, 1, 1], [5, 2, 3, 1])
        _renumber_projectile_resids(u, 3)
        assert list(u.residues.resids) == [5, 2, 3, 6]

    def test_no_projectile_unchanged(self):
        """When all atoms belong to the target, resids are not modified."""
        u = _make_universe([3, 3], [1, 2])
        _renumber_projectile_resids(u, 6)
        assert list(u.residues.resids) == [1, 2]

    def test_projectile_resids_contiguous(self):
        """Multiple projectile residues are numbered contiguously."""
        u = _make_universe([3, 3, 3, 3], [1, 99, 99, 99])
        _renumber_projectile_resids(u, 3)
        assert list(u.residues.resids) == [1, 2, 3, 4]

    def test_returns_same_universe(self):
        """Function modifies the universe in-place and returns it."""
        u = _make_universe([3, 3], [1, 1])
        result = _renumber_projectile_resids(u, 3)
        assert result is u
