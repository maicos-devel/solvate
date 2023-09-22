import MDAnalysis as mda
import numpy as np
from tqdm import tqdm

from .models import empty


def tile_universe(universe, n_x, n_y, n_z):
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


def pos_random(InsertionDomain):
    return np.array(
        np.random.rand(3) * (InsertionDomain[3:6] - InsertionDomain[0:3])
        + InsertionDomain[0:3],
        dtype=np.float32,
    )


def rot_random():
    u_1, u_2, u_3 = np.random.rand(3)

    theta, phi = np.arccos(2 * u_1 - 1), 2 * np.pi * u_2

    rot_vec = np.array(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )

    rot_angle = 360 * u_3

    return rot_angle, rot_vec


def SolvateCylinder(
    TargetUniverse,
    ProjectileUniverse,
    n=1,
    density=None,
    pos=None,
    radius=None,
    min=0,
    max=None,
    dim=2,
    distance=1.25,
    tries=1000,
    fudge_factor=1,
):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

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
        n = density * (2 * radius) ** 2 * (max - min)

    dimensionsTarget = TargetUniverse.dimensions.copy()

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms

    n = int(n)

    InsertionDomain = np.array(
        [pos[0] - radius, pos[1] - radius, min, pos[0] + radius, pos[1] + radius, max],
        dtype=np.float32,
    )

    InsertionVolume = (max - min) * np.pi * radius**2
    density = n / InsertionVolume

    SolvatedUniverse = SolvatePlanar(
        TargetUniverse, ProjectileUniverse, None, density, *InsertionDomain
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
            xmin,
            ymin,
            zmin,
            xmax,
            ymax,
            zmax,
            distance,
            solvate_factor,
            fudge_factor + 0.5,
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


def SolvatePlanar(
    TargetUniverse,
    ProjectileUniverse,
    n=1,
    density=None,
    xmin=0,
    ymin=0,
    zmin=0,
    xmax=None,
    ymax=None,
    zmax=None,
    distance=1.25,
    solvate_factor=100,
    fudge_factor=1,
    tries=1000,
):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

    # Use no fewer than 20 atoms for solvation
    SOLVATION_THRESHOLD = 20
    if density is not None:
        n = (
            density
            * TargetUniverse.dimensions[0]
            * TargetUniverse.dimensions[1]
            * TargetUniverse.dimensions[2]
        )

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

    InsertionDomain = [xmin, ymin, zmin, xmax, ymax, zmax]
    for i in np.arange(3):
        if InsertionDomain[i + 3] is None:
            InsertionDomain[i + 3] = TargetUniverse.dimensions[i]
    InsertionDomain = np.array(InsertionDomain)
    InsertionDomainSize = InsertionDomain[3:6] - InsertionDomain[0:3]
    dimensionsTarget = TargetUniverse.dimensions.copy()

    nAtomsTarget = TargetUniverse.atoms.n_atoms
    nAtomsProjectile = ProjectileUniverse.atoms.n_atoms
    n = int(n)
    print(f"Should solvate {n} Projectiles")
    x = np.ceil((n / (solvate_factor * fudge_factor)) ** (1 / 3)).astype(int)
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
    if n / (x**3) < SOLVATION_THRESHOLD and x > 2:
        x -= 1

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
    solvate_box_dimensions = np.concatenate(
        [InsertionDomainSize / x, dimensionsTarget[3:6]]
    )

    solvate_box = InsertPlanar(
        empty(solvate_box_dimensions),
        ProjectileUniverse,
        real_solvate_factor,
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
    TargetUniverse,
    ProjectileUniverse,
    n=1,
    xmin=0,
    ymin=0,
    zmin=0,
    xmax=None,
    ymax=None,
    zmax=None,
    distance=1.25,
    tries=1000,
):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

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

    for N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms

        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]
        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        for attempt in range(tries):
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
    TargetUniverse,
    ProjectileUniverse,
    n=1,
    pos=None,
    radius=None,
    min=0,
    max=None,
    dim=2,
    distance=1.25,
    tries=1000,
):
    """Insert the Projectile atoms into a cylindrical zone around
    TargetUniverse's center of geometry at random position and orientation
    and returns a new Universe."""

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

    for N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for attempt in range(tries):
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
    TargetUniverse,
    ProjectileUniverse,
    n=1,
    pos=None,
    radius=None,
    xmax=None,
    ymax=None,
    zmax=None,
    distance=1.25,
    tries=1000,
):
    """Inserts the Projectile atoms into a box in the TargetUniverse
    at random position and orientation and returns a new Universe."""

    def rand_spherical(radius=1):
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

    for N in tqdm(np.arange(n)):
        nAtomsTarget = TargetUniverse.atoms.n_atoms
        TargetUniverse = mda.Merge(TargetUniverse.atoms, ProjectileUniverse.atoms)
        TargetUniverse.dimensions = dimensionsTarget.copy()

        target = TargetUniverse.atoms[0:nAtomsTarget]
        projectile = TargetUniverse.atoms[-nAtomsProjectile:]

        ns = mda.lib.NeighborSearch.AtomNeighborSearch(target)

        # Generate coordinates and check for overlap
        for attempt in range(tries):
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
