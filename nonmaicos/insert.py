#!/usr/bin/env python3

import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

def pos_random(InsertionDomain):
    return np.array(np.random.rand(3)*(InsertionDomain[3:6] - InsertionDomain[0:3]) + InsertionDomain[0:3], dtype=np.float32)

def rot_random():
    u_1, u_2, u_3 = np.random.rand(3)

    theta = np.arccos(2*u_1 - 1)
    phi = 2*np.pi*u_2

    x = np.sin( theta ) * np.cos( phi )
    y = np.sin( theta ) * np.sin( phi )
    z = np.cos( theta )

    rot_vec = [x,y,z]
    rot_angle = 360*u_3

    return rot_angle, rot_vec 

def InsertCylinder(TargetUniverse, ProjectileUniverse, n=1, pos=None, radius=None, min=0, max=None, dim=2, distance=1.25, tries=1000):
    """Insert the Projectile atoms into a cylindrical zone in the
    around TargetUniverse's center of geometry at random position and orientation
    and returns a new Universe."""

    if max == None:
        max = TargetUniverse.dimensions[dim]

    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    if pos == None:
        if TargetUniverse.atoms.n_atoms == 0:
            pos = dimensionsTarget / 2
        else:
            pos = TargetUniverse.atoms.center_of_geometry()
    pos[dim] = min

    if radius == None:
        radius = np.min(dimensionsTarget) / 2

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

    for N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()
        
        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        for attempt in range(tries):   # Generate coordinates and check for overlap
            projectile.rotateby(*rot_random())

            r = radius * np.sqrt(np.random.rand())
            phi, z  = np.random.rand(2)* [2*np.pi, (max-min)]
            newcoord = np.roll([r*np.cos(phi),r*np.sin(phi), z],dim-2) + pos

            projectile.translate(newcoord - projectile.atoms.center_of_geometry())

            if len(ns.search(projectile,distance)) == 0:
                break
        else:
            raise RuntimeError("Error: No suitable position found, maybe you are trying to insert to many particles? Aborting.")

        projectile.residues.resids = projectile.residues.resids + target.residues.resids[-1]

    return TargetUniverse

def InsertPlanar(TargetUniverse, ProjectileUniverse, n=1, xmin=0, ymin=0, zmin=0, xmax=None, ymax= None, zmax=None, distance=1.25, tries=1000):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

    InsertionDomain = [xmin, ymin, zmin, xmax, ymax, zmax]
    InsertionDomain = InsertionDomain[0:3] + [TargetUniverse.dimensions[i] if InsertionDomain[i+3] == None else InsertionDomain[i+3] for i in np.arange(3)]
    InsertionDomain = np.array(InsertionDomain)
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    dimensionsTarget = TargetUniverse.dimensions.copy()

    ProjectileUniverse.atoms.translate(-ProjectileUniverse.atoms.center_of_geometry())

   
    if TargetUniverse.atoms.n_atoms == 0:
            TargetUniverse = ProjectileUniverse.copy()
            TargetUniverse.dimensions = dimensionsTarget
            TargetUniverse.atoms.translate(pos_random(InsertionDomain) - ProjectileUniverse.atoms.center_of_geometry())
            TargetUniverse.atoms.rotateby(*rot_random())
            n -= 1

    for N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms    

        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)        
        TargetUniverse.dimensions = dimensionsTarget

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]
        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        for attempt in range(tries):
            projectile.rotateby(*rot_random())
            projectile.translate(pos_random(InsertionDomain) - projectile.atoms.center_of_geometry())

            if len(ns.search(projectile,distance)) == 0:
                break
        else:
            raise RuntimeError("Error: No suitable position found, maybe you are trying to insert to many particles? Aborting.")

        projectile.residues.resids = projectile.residues.resids + target.residues.resids[-1]
        
    return TargetUniverse