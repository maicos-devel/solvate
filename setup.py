#!/usr/bin/env python
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2024 Authors and contributors
# (see the AUTHORS.rst file for the full list of names)
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later
"""Setup file for nonMAICoS package."""

from setuptools import setup

import versioneer


setup(
    name="nonmaicos",
    version=versioneer.get_version(),
    packages=["nonmaicos"],
)
