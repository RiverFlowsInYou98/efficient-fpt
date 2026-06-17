from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os
import sys


def _generate_defaults_pxi():
    """Read _defaults.py and write cython/_defaults.pxi with DEF constants."""
    defaults_py = os.path.join(
        os.path.dirname(__file__), "src", "efpt", "_defaults.py"
    )
    pxi_path = os.path.join(
        os.path.dirname(__file__), "src", "efpt", "cython", "_defaults.pxi"
    )
    namespace = {}
    with open(defaults_py) as f:
        exec(f.read(), namespace)

    lines = [
        "# Auto-generated from _defaults.py — do not edit manually.",
    ]
    for key, value in sorted(namespace.items()):
        if key.startswith("DEFAULT_"):
            lines.append(f"DEF {key} = {value!r}")
    lines.append("")  # trailing newline

    with open(pxi_path, "w") as f:
        f.write("\n".join(lines))


_generate_defaults_pxi()

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
        "efpt.cython.single_stage",  # no OpenMP
        sources=["src/efpt/cython/single_stage.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args,
    ),
    Extension(
        "efpt.cython.multi_stage",  # with OpenMP
        sources=["src/efpt/cython/multi_stage.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
    Extension(
        "efpt.cython.batch",  # with OpenMP
        sources=["src/efpt/cython/batch.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
    Extension(
        "efpt.cython.simulator",  # with OpenMP
        sources=["src/efpt/cython/simulator.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
    Extension(
        "efpt.cython.utils",  # with OpenMP (for omp_get_max_threads)
        sources=["src/efpt/cython/utils.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=base_compile_args + omp_compile_args,
        extra_link_args=omp_link_args,
    ),
]

setup(
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
