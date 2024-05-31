#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

dist = setup(
    name="dckernel",
    version="1.0.0",
    description="Robust Kernel Testing under Data Corruption",
    author="Antonin Schrab",
    author_email="a.lastname@ucl.ac.uk",
    license="MIT License",
    packages=["dckernel", ],
    install_requires=["jax", "jaxlib"],
    python_requires=">=3.9",
)
