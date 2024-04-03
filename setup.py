"""Inspired by https://github.com/Eclectic-Sheep/sheeprl/blob/main/setup.py"""

import pathlib

from setuptools import setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the aevb version."""
    path = CWD / "aevb" / "__version__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("Bad version data in __version__.py")


setup(name="aevb", version=get_version())
