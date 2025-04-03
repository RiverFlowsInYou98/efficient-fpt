from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import sys

# Platform-specific OpenMP flags
if sys.platform == "darwin":
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = ["-lomp"]
else:
    omp_compile_args = ["-fopenmp"]
    omp_link_args = ["-fopenmp"]

base_compile_args = ["-O3", "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION"]

extensions = [
    Extension(
        "efficient_fpt.single_stage_cy",  # no OpenMP
        sources=["src/efficient_fpt/single_stage_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args,
    ),
    Extension(
        "efficient_fpt.multi_stage_cy",  # with OpenMP
        sources=["src/efficient_fpt/multi_stage_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
]

setup(
    name="efficient_fpt",
    version="0.1",
    packages=["efficient_fpt"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
)
