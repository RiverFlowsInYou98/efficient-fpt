# multi_stage.pxd — declarations for cimport by batch.pyx

cdef double _compute_addm_logfptd_core(
    double t, int d,
    double[:] mu_array, double[:] sacc_array,
    double sigma, double a, double b, double x0, int bdy,
    int order_mid, double[:] x_ref_mid_in, double[:] w_ref_mid_in,
    int order_last, double[:] x_ref_last_in, double[:] w_ref_last_in,
    int trunc_num, double threshold,
    bint log_space,
) noexcept nogil
