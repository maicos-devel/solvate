#!/usr/bin/env python3

import MDAnalysis as mda
import numpy as np
import warnings


def empty(dimensions):

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Residues specified but no atom_resindex given.  All atoms will be placed in first Residue.")
        warnings.filterwarnings("ignore", message="Segments specified but no segment_resindex given.  All residues will be placed in first Segment")
        u = mda.Universe.empty(0, trajectory=True)
    
    u.dimensions = dimensions
    return u

def spce():

    l_1 = 1
    q_O = -0.8476
    q_H = q_O / 2
    theta = np.deg2rad(109.47)

    return type_a(l_1, q_O, q_H, theta)


def tip4p_epsilon():

    l_1 = 0.9572
    l_2 = 0.105

    q_H = 0.5270
    q_M = -1.054

    theta = np.deg2rad(104.52)

    return type_c(l_1, l_2, q_M, q_H, theta)

def type_a(l_1, q_O, q_H, theta):

    model = mda.Universe.empty(3,
                         n_residues=1,
                         atom_resindex=[0,0,0],
                         residue_segindex=[0],
                         trajectory=True)

    model.add_TopologyAttr('name', ['OW', 'HW1', 'HW2'])
    model.add_TopologyAttr('type', ['O', 'H', 'H'])
    model.add_TopologyAttr('resname', ['SOL'])
    model.add_TopologyAttr('resid', [1])
    model.add_TopologyAttr('segid', ['SOL'])
    model.add_TopologyAttr('charges', [q_O, q_H, q_H])

    pos_O = np.array([ 0,        0,       0      ], dtype=np.float32)
    pos_H1 = pos_O + np.array([ 0,        l_1,       0      ], dtype=np.float32)
    pos_H2 = pos_O + np.array([ l_1 * np.cos(theta - np.pi/2), l_1 * np.sin(theta - np.pi/2), 0], dtype=np.float32)

    model.atoms.positions = np.array([pos_O, pos_H1, pos_H2])

    return model


def type_c(l_1, l_2, q_M, q_H, theta):

    model = mda.Universe.empty(4,
                            n_residues=1,
                            atom_resindex=[0,0,0],
                            residue_segindex=[0],
                            trajectory=True)

    model.add_TopologyAttr('name', ['OW', 'HW1', 'HW2', 'MW'])
    model.add_TopologyAttr('type', ['O', 'H', 'H', 'D'])
    model.add_TopologyAttr('resname', ['SOL'])
    model.add_TopologyAttr('resid', [1])
    model.add_TopologyAttr('segid', ['SOL'])
    model.add_TopologyAttr('charges', [0, q_H, q_H, q_M])


    pos_O = np.array([ 0,        0,       0      ], dtype=np.float32)
    pos_H1 = pos_O + np.array([ 0,        l_1,       0      ], dtype=np.float32)
    pos_H2 = pos_O + np.array([ l_1 * np.cos(theta - np.pi/2), l_1 * np.sin(theta - np.pi/2), 0], dtype=np.float32)
    pos_M = pos_O + np.array([ l_2 * np.cos(np.pi/2 - theta/2), l_2 * np.sin(np.pi/2 - theta/2), 0], dtype=np.float32)

    model.atoms.positions = np.array([pos_O, pos_H1, pos_H2, pos_M])

    return model