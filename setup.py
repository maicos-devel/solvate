from setuptools import setup

import versioneer

setup(
    name="nonmaicos",
    version=versioneer.get_version(),
    packages=["nonmaicos"],
    install_requires=[
        'MDAnalysis>=2.0.0',
        'numpy'
        ],
    )
