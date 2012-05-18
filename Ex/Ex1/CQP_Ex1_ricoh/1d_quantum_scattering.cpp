/*
 * 1D quantum scattering problem
 * for CQP FS 2012
 * by Rico Häuselmann
 */

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

typedef std::pair<double, double> ab_pair;

double numerov(double phi_n, double phi_n_minus_1, double k_n_plus_1, double k_n, double k_n_minus_1, double delta_x)
{
    return ((2 * (1 - (5 * delta_x * delta_x / 12 * k_n)) * phi_n) - 
            ((1 + (delta_x * delta_x / 12 * k_n_minus_1)) * phi_n_minus_1)) / 
            (1 + (delta_x * delta_x / 12 * k_n_plus_1));
}

ab_pair get_AB(double E, double delta_x, double a)
{
    int End = a / delta_x;

    double phi_a = 1;
    double phi_a_plus_delta_x = std::exp(-delta_x * std::sqrt(2 * E));

    double phi_n = phi_a;
    double phi_n_minus_1 = phi_a_plus_delta_x;

    double k_n = 2 * (E - 1);
    double k_n_minus_1 = 2 * E;
    double k_n_plus_1 = k_n;

    double phi_n_plus_1 = numerov(phi_n, phi_n_minus_1, k_n_plus_1, k_n, k_n_minus_1, delta_x);
    phi_n_minus_1 = phi_n;
    phi_n = phi_n_plus_1;
    k_n_minus_1 = k_n;

    for(int i=0; i<End-1; ++i)
    {
        phi_n_plus_1 = numerov(phi_n, phi_n_minus_1, k_n_plus_1, k_n, k_n_minus_1, delta_x);
        phi_n_minus_1 = phi_n;
        phi_n = phi_n_plus_1;
        //std::cout << phi_n << std::endl;
    }

    k_n_plus_1 = 2 * E;
    phi_n_plus_1 = numerov(phi_n, phi_n_minus_1, k_n_plus_1, k_n, k_n_minus_1, delta_x);
    phi_n_minus_1 = phi_n;
    phi_n = phi_n_plus_1;

    double phi_0 = phi_n;
    double A = (phi_0 - (phi_0 / k_n)) / 2;
    double B = phi_0 - A;

    ab_pair result(A, B);
    return result;
}

int main (int argc, char const* argv[])
{
    double E = 0.1;
    double delta_x = 0.0001;
    double a = 1;

    ab_pair AB = get_AB(E, delta_x, a);
    std::cout << "A: " << AB.first << " "
              << "B: " << AB.second << " "
              << "dx: " << delta_x << std::endl;
    
    // observe tunneling
    std::vector<double> tunnel_run_E;
    std::vector<ab_pair> tunnel_run_ab;
    double n_energies = 100;
    for(int i = 0; i < n_energies; ++i)
    {
        E = i / n_energies;
        tunnel_run_E.push_back(E);
        tunnel_run_ab.push_back(get_AB(E, delta_x, a));
        //std::cout << "A: " << tunnel_run_ab.back().first << " "
        //          << "B: " << tunnel_run_ab.back().second << " "
        //          << "E: " << E << std::endl;
    }

    // check stability wrt to timestep
    E = 0.2;
    std::vector<double> delta_run_dx;
    std::vector<ab_pair> delta_run_ab;
    for(int i = 1; i < 25; ++i)
    {
        delta_x = 1.0/(1<<i);
        delta_run_dx.push_back(delta_x);
        delta_run_ab.push_back(get_AB(E, delta_x, a));
        //std::cout << "A: " << delta_run_ab.back().first << " "
        //          << "B: " << delta_run_ab.back().second << " "
        //          << "dx: " << delta_x << std::endl;
    }

    // T dependency on a
    delta_x = 0.00001;
    std::vector<double> t_of_a_run_a;
    std::vector<ab_pair> t_of_a_run_ab;
    for(int i = 0; i < 100; ++i)
    {
        a = i / 10.;
        t_of_a_run_a.push_back(a);
        t_of_a_run_ab.push_back(get_AB(E, delta_x, a));
        //std::cout << "A: " << delta_run_ab.back().first << " "
        //          << "B: " << delta_run_ab.back().second << " "
        //          << "a: " << a << std::endl;
    }

    // write out plot file in format
    // (1) #E #T 
    std::ofstream plot1("tunnel.dat");
    for(int i = 0; i < tunnel_run_E.size(); ++i)
    {
        plot1 << tunnel_run_E[i] << " " 
              << 1 / (tunnel_run_ab[i].first * tunnel_run_ab[i].first) 
              << std::endl;
    }
    // (2) #delta_x #A #B
    std::ofstream plot2("delta.dat");
    for(int i = 0; i < delta_run_dx.size(); ++i)
    {
        plot2 << delta_run_dx[i] << " " 
              << delta_run_ab[i].first << " " 
              << delta_run_ab[i].second 
              << std::endl;
    }
    // (3) #a #T
    std::ofstream plot3("Tdep.dat");
    for(int i = 0; i < t_of_a_run_a.size(); ++i)
    {
        plot3 << t_of_a_run_a[i] << " " 
              << 1 / (t_of_a_run_ab[i].first * t_of_a_run_ab[i].first) 
              << std::endl;
    }

    return 0;
}
