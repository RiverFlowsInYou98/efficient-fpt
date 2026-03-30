# multi_stage.pxd — declarations for cimport by batch.pyx

cdef double _compute_addm_fptd_core(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int order, double[:] x_ref_in, double[:] w_ref_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil
