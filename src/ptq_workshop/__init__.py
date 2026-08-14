"""Reproducible NVIDIA Blackwell post-training quantization workshop."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("blackwell-ptq-workshop")
except PackageNotFoundError:
    __version__ = "0.1.0"
