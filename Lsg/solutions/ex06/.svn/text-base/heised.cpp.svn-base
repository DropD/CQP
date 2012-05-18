#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <alps/utility/bitops.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/random.hpp>
#include <boost/tuple/tuple.hpp>

class SpinHalfBasis
{
public:
    typedef unsigned State;
    typedef unsigned Index;
    
    SpinHalfBasis(unsigned l, unsigned nups);
    
    State state( Index i ) const { return states_[i]; }
    Index index( State s ) const { return index_[s]; }
    Index dimension() const { return states_.size(); }
    
private:
    std::vector<State> states_;
    std::vector<Index> index_;
};

SpinHalfBasis::SpinHalfBasis(unsigned l, unsigned nups)
:   index_(State(1<<l),std::numeric_limits<Index>::max())
{
    // find all states with [nups] up spins (i.e. total magnetization sz=nups-l/2)
    for( State s = 0; s < index_.size(); ++s )
    {
        if( alps::popcnt(s) == nups )
        {
            index_[s] = states_.size();
            states_.push_back(s);
        }
    }
}

class HeisenbergHamiltonian : public SpinHalfBasis
{
public:
	typedef double Scalar;
	typedef boost::numeric::ublas::vector<Scalar> Vector;
    
    HeisenbergHamiltonian(unsigned l, unsigned nups, bool periodic, double j);
    
    void multiply(const Vector& x, Vector& y) const;
    
private:
    unsigned l_;                // length of the chain
    bool periodic_;             // periodic boundary conditions off/on
    double j_;                  // coupling J
};

HeisenbergHamiltonian::HeisenbergHamiltonian(unsigned l, unsigned nups, bool periodic, double j)
:   SpinHalfBasis(l,nups)
,   l_(l)
,   periodic_(periodic)
,   j_(j)
{
}

/// Calculate y = H x
void HeisenbergHamiltonian::multiply(const Vector& x, Vector& y) const
{
    // check dimensions
    assert( x.size() == dimension() );
    assert( y.size() == dimension() );
    
    // diagonal part: +J/4 for parallel, -J/4 for antiparallel neighbors
    State mask = (1<<(l_-1)) - 1;
    for( Index i = 0; i < dimension(); ++i )
    {
        State s = state(i);
        int p = alps::popcnt(mask & ( s ^ (~s>>1) )); // number of parallel pairs
        y[i] = .25*j_*( 2.*p - l_ + 1 )*x[i];
    }
    
    // off-diagonal part: {01,10} -> J/2 {10,01}
    for( Index i = 0; i < dimension(); ++i )
    {
        State s = state(i);
        for( int r = 0; r < l_-1; ++r )
        {
            State sflip = s ^ (3<<r);
            Index j = index(sflip);
            if( j < dimension() )
                y[j] += .5*j_*x[i];
            }
    }
    
    // periodic boundaries
    if( periodic_ )
    {
        for( Index i = 0; i < dimension(); ++i )
        {
            State s = state(i);
            // diagonal
            int p = 1 & ( s ^ (~s>>(l_-1)) );
            y[i] += .25*j_*( 2.*p - 1 )*x[i];
            // off-diagonal
            State sflip = s ^ ( 1 | (1<<(l_-1)) );
            Index j = index(sflip);
            if( j < dimension() )
                y[j] += .5*j_*x[i];
        }
    }
}

namespace ietl
{
	inline void mult( const HeisenbergHamiltonian& h, const HeisenbergHamiltonian::Vector& x, HeisenbergHamiltonian::Vector& y )
	{
		h.multiply( x, y );
	}
}

// ietl::mult() needs to be declared before including these
#include <ietl/interface/ublas.h>
#include <ietl/lanczos.h>

std::pair< std::vector<double>, std::vector<int> >
diagonalize( const HeisenbergHamiltonian& h, unsigned nvals=1, unsigned maxiter=1000)
{
	typedef ietl::vectorspace<HeisenbergHamiltonian::Vector> Vectorspace;
	typedef ietl::lanczos<HeisenbergHamiltonian,Vectorspace> Lanczos;
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

void do_chain(int l, bool periodic, double j, unsigned nstates, std::ostream& datfile)
{
    std::cout << "+++ Diagonalizing S=1/2 Heisenberg " << (periodic ? "periodic" : "open")
        << " chain with L=" << l << ", J=" << j << std::endl;
    
    std::vector<double> all_energies;
    
    for( unsigned n = 0; n <= l; ++n )
    {
        double sz = n-l/2.;
        std::cout << "--- Sector Sz=" << sz << ": ";
        HeisenbergHamiltonian ham(l,n,periodic,j);
        std::cout << ham.dimension() << " basis states" << std::endl;
     
        unsigned nvals = std::min(nstates,ham.dimension());   
        std::vector<double> evals;
        std::vector<int> mults;
        boost::tie(evals,mults) = diagonalize(ham,nvals);
        
        for( unsigned i = 0; i < nvals; ++i )
            std::cout << n-l/2. << '\t' << evals[i] << '\t' << mults[i] << '\n';
        all_energies.insert(all_energies.end(),evals.begin(),evals.begin()+nvals);
    }
    
    std::sort(all_energies.begin(),all_energies.end());
    datfile << l;
    for( unsigned i = 0; i < nstates; ++i )   datfile << '\t' << all_energies[i];
    datfile << '\n';
    
}

void write_dat_header(std::ostream& datfile,unsigned nstates)
{
    datfile << "# Sz";
    for( unsigned i = 0; i < nstates; ++i )
        datfile << "\tE" << i;
    datfile << "\n";
    datfile.precision(10);
}

int main()
{
    int lmin = 2;
    int lmax = 20;
    double j = 1.;
    unsigned nstates = 2; // How many of the lowest eigenvalues (energies) do you want to calculate? ( nstates = 1 ... just the ground state)

    std::ofstream opendata("openchain.dat");
    write_dat_header(opendata,nstates);
    for( int l = lmin; l <= lmax; ++l )
        do_chain(l,false,j,nstates,opendata);

    std::ofstream periodicdata("periodicchain.dat");
    write_dat_header(periodicdata,nstates);
    for( int l = lmin; l <= lmax; ++l )
        do_chain(l,true,j,nstates,periodicdata);
}
