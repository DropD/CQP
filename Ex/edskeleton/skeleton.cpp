#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/random.hpp>
#include <boost/tuple/tuple.hpp>

class Hamiltonian
{
public:
	typedef double Scalar;
	typedef boost::numeric::ublas::vector<Scalar> Vector;
    
    /// Calculate y = H x
    void multiply(const Vector& x, Vector& y) const;

    // ...    
};

namespace ietl
{
	inline void mult( const Hamiltonian& h, const Hamiltonian::Vector& x, Hamiltonian::Vector& y )
	{
		h.multiply( x, y );
	}
}

// ietl::mult() needs to be declared before including these
#include <ietl/interface/ublas.h>
#include <ietl/lanczos.h>

std::pair< std::vector<double>, std::vector<int> >
diagonalize( const Hamiltonian& h, unsigned nvals=1, unsigned maxiter=1000)
{
	typedef ietl::vectorspace<Hamiltonian::Vector> Vectorspace;
	typedef ietl::lanczos<Hamiltonian,Vectorspace> Lanczos;
    typedef ietl::lanczos_iteration_nlowest<double> Iteration;
    typedef boost::mt19937 Generator;
	
    Vectorspace vspace(h.dimension());
    Lanczos solver(h,vspace);
	Iteration iter(maxiter,nvals);
	solver.calculate_eigenvalues(iter,Generator());
	
	if( iter.iterations() == maxiter )
        std::cerr << "Lanczos did not converge in " << iter.iterations() << " iterations." << std::endl;
    else
        std::cout << "  Lanczos converged after " << iter.iterations() << " iterations." << std::endl;

    return std::make_pair(solver.eigenvalues(),solver.multiplicities());
}

