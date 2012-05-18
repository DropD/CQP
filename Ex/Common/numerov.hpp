#ifndef CQP_NUMEROV
#define CQP_NUMEROV

namespace cqp
{
    template<typename T>
    class numerov
    {
        private:
        T dx;
        T dx_fac;

        public:
        numerov(double dx) : dx(dx), dx_fac(dx * dx / 12) {}

        template<typename F>
        T step(T x, F k, T psi_nm1, T psi_n) 
        {
            int n = 1;
            T a = (2 - (k(x) * 10 * dx_fac)) * psi_n;
            T b = (1 + (k(x - dx) * dx_fac)) * psi_nm1;
            T c = (1 + (k(x + dx) * dx_fac));
            return (a - b) / c;
        }

        template<typename F>
        T integrate(T x_begin, T x_end, F k, T psi_0, T psi_1)
        {
            T psi_nm1 = psi_0;
            T psi_n   = psi_1;
            int n = (x_end - x_begin) / dx;
            for(int i = 0; i < n; ++i)
            {
                T xn = x_begin + i*dx;
                T psi_np1 = step(xn, k, psi_nm1, psi_n);
                psi_nm1 = psi_n;
                psi_n = psi_np1;
            }
            return psi_n;
        }
    };
}


#endif
