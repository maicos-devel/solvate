#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the nonMAICoS package."""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import nonmaicos


class TestInserts(object):
    """Test the insertion code."""

    @pytest.mark.parametrize("n_water", (1, 10, 100))
    def test_insert_planar_n_water(self, n_water):
        """Test the number of inserted particles in InsertPlanar."""
        emptyUniverse = nonmaicos.models.empty([20, 20, 20, 90, 90, 90])
        testParticle = nonmaicos.models.spce()
        u = nonmaicos.InsertSphere(emptyUniverse, testParticle, n_water)
        assert u.atoms.n_atoms == n_water * 3

    # TODO: def test_insert_planar_density(self):
    #     """Test the density of the inserted particles in InsertPlanar."""

    # TODO: def test_insert_sphere_n_water(self):
    #     """Test the number of inserted particles in InsertSphere."""

    # TODO: def test_insert_sphere_density(self):
    #     """Test the density of the inserted particles in InsertSphere."""

    # TODO: def test_insert_cylinder_n_water(self):
    #     """Test the number of inserted particles in InsertCylinder."""

    # TODO: def test_insert_cylinder_density(self):
    #     """Test the density of the inserted particles in InsertCylinder."""

    # TODO: def test_insert_planar_domain(self):
    #     """Test the domain of the inserted particles in InsertPlanar."""

    # TODO: def test_insert_sphere_domain(self):
    #     """Test the domain of the inserted particles in InsertSphere."""

    # TODO: def test_insert_cylinder_domain(self):
    #     """Test the domain of the inserted particles in InsertCylinder."""


# class TestSolvate(object):
#     """Test the solvation code."""

# TODO: test_solvate_planar_n_water(self):
#     """Test the solvation of a planar system."""

# TODO: test_solvate_sphere_n_water(self):
#     """Test the solvation of a spherical system."""

# TODO: test_solvate_cylinder_n_water(self):
#     """Test the solvation of a cylindrical system."""

# TODO: test_solvate_planar_density(self):
#     """Test the density of the solvated system."""

# TODO: test_solvate_sphere_density(self):
#     """Test the density of the solvated system."""

# TODO: test_solvate_cylinder_density(self):
#     """Test the density of the solvated system."""


class TestModels(object):
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
        u = nonmaicos.models.type_a(1, -2, 0.4238, np.deg2rad(109.47))
        assert u.atoms.n_atoms == 3
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1

    @pytest.mark.parametrize("angle", (45, 90, 125, 180))
    def test_three_site_angle(self, angle):
        """Test three site model build function."""
        pos_O, pos_H1, pos_H2 = nonmaicos.models._three_site_molecule(np.deg2rad(angle))

        # We expect the middle atom to be at the origin for now.
        assert_allclose(pos_O, np.array([0.0, 0.0, 0.0]))
        # Make sure the vectors are normalized (ignoring floating point errors)
        assert_allclose(np.linalg.norm(pos_H1), 1.0)
        assert_allclose(np.linalg.norm(pos_H2), 1.0)
        # Calculate the angle
        alpha = np.arccos(np.dot(pos_H1, pos_H2))
        # If the angle is correct, we are happy
        assert_allclose(np.rad2deg(alpha), angle)

    @pytest.mark.parametrize("angle", (0, 181, -1))
    def test_three_site_error(self, angle):
        """Test three site model ValueError."""
        with pytest.raises(ValueError):
            nonmaicos.models._three_site_molecule(np.deg2rad(angle))

    @pytest.mark.parametrize("angle", (45, 90, 125, 180))
    def test_type_a_angle(self, angle):
        """Test type a angle."""
        u = nonmaicos.models.type_a(1, -2, 1, np.deg2rad(angle))
        assert_allclose(u.atoms.angles[0].value(), angle)

    def test_type_c_general(self):
        """Test type c."""
        """Type c water molecules should have 4 atoms, 2 bonds, and 1 angle."""
        u = nonmaicos.models.type_c(0.9572, 0.105, -1.054, 0.527, np.deg2rad(104.52))
        assert u.atoms.n_atoms == 4
        assert len(u.atoms.bonds) == 2
        assert len(u.atoms.angles) == 1

    @pytest.mark.parametrize("angle", (45, 90, 125, 180))
    def test_type_c_angle(self, angle):
        """Test type c angle."""
        u = nonmaicos.models.type_c(0.9572, 0.105, -1.054, 0.527, np.deg2rad(angle))
        assert_allclose(u.atoms.angles[0].value(), angle)

    def test_spce(self):
        """Test SPC/E water.

        This is a regression test, the values here should NEVER change.
        """
        # TODO: Check the values against a source, this is a regression test
        u = nonmaicos.models.spce()
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
        u = nonmaicos.models.tip4p_epsilon()
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
