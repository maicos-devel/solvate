# solvate: A simple tool to create solvated molecular systems in python

solvate is a lightweight Python package designed to quickly build initial structures of
molecular dynamics simulations. Based on MDAnalysis, it is flexible and easy to use.

This tool is inspired by similar tools like GROMACS `solvate`, but aims to provide a more user-friendly interface and additional features.


# Features

- Insert molecules into target systems of different geometries (cubic, rectangular, spherical)
- Fast solvation of large structures using spatial partitioning
- Customizable solvent and solute molecules
- Support for various file formats via MDAnalysis

# Installation

You can install solvate via pip:

```bash
pip install solvate
```
Or clone the repository and install it manually:

```bash
git clone https://github.com/maicos-devel/solvate.git
cd solvate
pip install .
```

# Usage

See the [documentation](https://maicos-devel.github.io/solvate/) for detailed usage instructions and examples.

Also check out the [examples](https://github.com/maicos-devel/solvate/tree/main/examples) directory for practical use cases.

# Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) for more information.
Feel free to open issues and submit pull requests.

# Similar Software

- [packmol](https://github.com/m3g/packmol)
- [mdapackmol](https://github.com/MDAnalysis/MDAPackmol)
- [moltemplate](https://moltemplate.org/)
- [autosolvate](https://autosolvate.readthedocs.io/en/latest/)