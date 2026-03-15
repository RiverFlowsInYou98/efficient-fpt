from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import subprocess
import sys

# Platform-specific OpenMP flags
omp_include_dirs = []
omp_library_dirs = []

if sys.platform == "darwin":
    omp_compile_args = ["-Xpreprocessor", "-fopenmp"]
    omp_link_args = ["-lomp"]
    try:
        omp_prefix = subprocess.check_output(
            ["brew", "--prefix", "libomp"], text=True
        ).strip()
        omp_include_dirs = [omp_prefix + "/include"]
        omp_library_dirs = [omp_prefix + "/lib"]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
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
        include_dirs=[numpy.get_include()] + omp_include_dirs,
        library_dirs=omp_library_dirs,
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
    Extension(
        "efficient_fpt.addm_simulator_cy",  # no OpenMP
        sources=["src/efficient_fpt/addm_simulator_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args,
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
            "initializedcheck": False,
        },
    ),
)
