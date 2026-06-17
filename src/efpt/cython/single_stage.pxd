# single_stage.pxd

cpdef double fptd_basic(double t, double mu, double a1, double b1,
                        double a2, double b2, int bdy,
                        int trunc_num=*, double threshold=*,
                        bint adaptive_stopping=*) noexcept nogil

cpdef double q_basic(double x, double mu, double a1, double b1,
                     double a2, double b2, double T,
                     int trunc_num=*, double threshold=*,
                     bint adaptive_stopping=*) noexcept nogil

cpdef double fptd_single(double t, double mu, double sigma,
                         double a1, double b1, double a2, double b2,
                         double x0, int bdy,
                         int trunc_num=*, double threshold=*,
                         bint adaptive_stopping=*) noexcept nogil

cpdef double q_single(double x, double mu, double sigma,
                      double a1, double b1, double a2, double b2,
                      double T, double x0,
                      int trunc_num=*, double threshold=*,
                      bint adaptive_stopping=*) noexcept nogil
