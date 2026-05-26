CHANGELOG file
--------------

The rules for solvate's CHANGELOG file:

- entries are sorted newest-first.
- summarize sets of changes (don't reproduce every git log comment here).
- don't ever delete anything.
- keep the format consistent (79 char width, Y/M/D date format) and do not
  use tabs but use spaces for formatting

.. inclusion-marker-changelog-start

Unreleased
----------

v0.1.0 (2026/05/26)
-------------------
Henrik Stooß, Philip Loche, Kira Fischer

- Make projectile residues always monotonic (#19)
- Make the documentation more comprehensive and user-friendly (#20)
- Add .readthedocs.yml to manifest (#18)
- Add documentation and README.md (#10)
- Add CI (#2)
- Rename nonMAICoS to solvate, move repository to GitHub
- Fix wrong angles in water models (!6)
- Add CI, linting, and testing (!6)
- Fix calculation of the number of particles from the density (!5)
- Fixed a bug in calling SolvatePlanar (!3)
- Fix pbc handling (!2)

v0.0.4 (2023/04/14)
-------------------
Henrik Jaeger

- Add SolvatePlanar

v0.0.3 (2022/11/02)
-------------------
Henrik Jaeger

- Initial version

.. inclusion-marker-changelog-end
