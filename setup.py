#!/bin/env python3

"""Python helper code for the model runtime."""

import logging
import platform
from setuptools import setup, find_packages  # type: ignore


def list_all_deps():
    with open("requirements.txt") as f:
        return f.readlines()


setup(
    name="model_helpers",
    version="0.1",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=list_all_deps(),
)
