"""Cython utility functions for efficient-fpt."""

from libc.math cimport log, INFINITY, NAN
from openmp cimport omp_get_max_threads


cdef inline double positive_log(double value) noexcept nogil:
    """Return log(x) for x > 0, -inf for x <= 0, and nan when x is nan."""
    if value > 0.0:
        return log(value)
    if value != value:
        return NAN
    return -INFINITY


cpdef double positive_log_wrapper(double value):
    """Python-facing wrapper used to test the Cython scalar positive-log helper."""
    return positive_log(value)


cpdef print_num_threads():
    """Print the number of available OpenMP threads."""
    print("Number of available threads:", omp_get_max_threads())
