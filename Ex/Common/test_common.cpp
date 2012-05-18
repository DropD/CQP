#include "numerov.hpp"

#include <iostream>

template<typename F, typename T>
struct k_factor
{
    k_factor(F Potential, T Energy) : V(Potential), E(Energy) {}
    T operator() (T x) { return 2 * (E - V(x)); }
    F V;
    T E;
};

template<typename T>
struct rectangular_potential
{
    T a, b;
    rectangular_potential(T a, T b) : a(a), b(b) {}
    T operator() (T x) { return (x >= a && x < b) ? 1 : 0; }
};

int main (int argc, char const* argv[])
{
    double a = 10;
    double dx = -0.001;
    double E = 0.5;
    rectangular_potential<double> V(0, a);
    k_factor<rectangular_potential<double>, double> k(V, 0.5);
    cqp::numerov<double> integrator(dx);
    std::cout << integrator.integrate(10, 0, k, 0.7, 1) << std::endl;
    
    return 0;
}
