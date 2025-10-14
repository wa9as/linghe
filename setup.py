# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import pathlib

import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in
                        pkg_resources.parse_requirements(f)]

setup(
    name="tflops",
    version="0.0.2",
    license="MIT",
    license_files=("LICENSE",),
    description="flood ops",
    URL="https://code.alipay.com/pia/flops",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
)
