# single_stage_cy.pxd

cpdef double fptd_basic_cy(double t, double mu, double a1, double b1,
                            double a2, double b2, int bdy,
                            int trunc_num=*, double threshold=*) noexcept nogil

cpdef double q_basic_cy(double x, double mu, double a1, double b1,
                        double a2, double b2, double T,
                        int trunc_num=*, double threshold=*) noexcept nogil

cpdef double fptd_single_cy(double t, double mu, double sigma,
                            double a1, double b1, double a2, double b2,
                            double x0, int bdy,
                            int trunc_num=*, double threshold=*) noexcept nogil

cpdef double q_single_cy(double x, double mu, double sigma,
                         double a1, double b1, double a2, double b2,
                         double T, double x0,
                         int trunc_num=*, double threshold=*) noexcept nogil
