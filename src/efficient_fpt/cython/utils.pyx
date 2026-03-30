"""Cython utility functions for efficient-fpt."""

from openmp cimport omp_get_max_threads


cpdef print_num_threads():
    """Print the number of available OpenMP threads."""
    print("Number of available threads:", omp_get_max_threads())
