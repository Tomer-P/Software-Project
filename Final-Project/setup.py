# Build the Python extension from the C wrapper and matrix implementation.
from setuptools import setup, Extension

module = Extension(
    "symnmfmodule",
    sources=["symnmfmodule.c", "symnmf.c"]
)

setup(
    name="symnmfmodule",
    version="1.0",
    ext_modules=[module]
)
