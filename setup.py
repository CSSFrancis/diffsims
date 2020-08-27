# -*- coding: utf-8 -*-
# Copyright 2017-2020 The diffsims developers
#
# This file is part of diffsims.
#
# diffsims is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# diffsims is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with diffsims.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages

<<<<<<< HEAD
exec(open("diffsims/release_info.py").read())  # grab version info
=======
exec(open('diffsims/release_info.py').read())  # grab version info
>>>>>>> 240204350ad4261d80b52bd267ca0eaf9d4703fc


setup(
    name=name,
    version=version,
<<<<<<< HEAD
    description="Diffraction Simulations in Python.",
=======
    description='Diffraction Simulations in Python.',
>>>>>>> 240204350ad4261d80b52bd267ca0eaf9d4703fc
    author=author,
    author_email=email,
    license=license,
    url="https://github.com/pyxem/diffsims",
    long_description=open("README.rst").read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(),
    # adjust the tabbing
    install_requires=[
        "scipy>=0.15",
        "numpy>=1.10",
        "matplotlib>=3.1.1",
        "tqdm>=0.4.9",
        "transforms3d",
        "diffpy.structure>=3.0.0",  # First Python 3 support
        "orix>=0.4.0",
        "numba",
        "psutil",
    ],
    python_requires=">=3.6",
    package_data={"": ["LICENSE", "readme.rst",], "diffsims": ["*.py"],},
)
