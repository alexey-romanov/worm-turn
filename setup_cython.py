import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("longest_chain_cy.pyx", annotate=True, build_dir="cythonize_build",), include_dirs=[numpy.get_include()],
)
