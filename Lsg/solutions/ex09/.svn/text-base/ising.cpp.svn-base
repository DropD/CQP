/*****************************************************************************
 *
 * Copyright (C) 2010 - 2012 Jan Gukelberger
 * Copyright (C) 2010 Brigitte Surer
 *
 * This software is based on the ALPS Code-02 tutorial, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <alps/scheduler/montecarlo.h>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <boost/multi_array.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
using namespace std;


class Simulation
{
public:
    Simulation(double Jy, double Jx, size_t M, size_t L, std::string output_file)
    :   eng_(42)
    ,   rng_(eng_, dist_)
    ,   spins_(boost::extents[M][L])
    ,   energy_("E")
    ,   magnetization_("m")
    ,   abs_magnetization_("|m|")
    ,   m2_("m^2")
    ,   m4_("m^4")
    ,   filename_(output_file)
    {
        L_[0] = M ; L_[1] = L ;
        J_[0] = Jy; J_[1] = Jx;
        exp2j_[0] = exp(-2*Jy);
        exp2j_[1] = exp(-2*Jx);
        
        // Init random spin configuration
        for(unsigned i = 0; i < L_[0]; ++i)
        {
            for(unsigned j = 0; j < L_[1]; ++j)
                spin(make_pair(i,j)) = 2 * randint(2) - 1;
        }
    }

    void run(size_t ntherm,size_t n)
    {
        thermalization_ = ntherm;
        sweeps_=n;
        // Thermalize for ntherm steps
        while(ntherm--)
            step();

        // Run n steps
        while(n--)
        {
            step();
            measure();
        }
        
        //save the observables 
        save(filename_);
        
        // Print observables
        // std::cout << abs_magnetization_;
        std::cout << abs_magnetization_.name() << ":\t" << abs_magnetization_.mean()
            << " +- " << abs_magnetization_.error() << ";\ttau = " << abs_magnetization_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(abs_magnetization_.converged_errors()) 
            << std::endl;
        std::cout << energy_.name() << ":\t" << energy_.mean()
            << " +- " << energy_.error() << ";\ttau = " << energy_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(energy_.converged_errors()) 
            << std::endl;
        std::cout << magnetization_.name() << ":\t" << magnetization_.mean()
            << " +- " << magnetization_.error() << ";\ttau = " << magnetization_.tau() 
            << ";\tconverged: " << alps::convergence_to_text(magnetization_.converged_errors())
            << std::endl;
    }
    
    void step()
    {
        // printSpins();
        vector<TodoItem> todo;
        set<Site> flipped;
        
        // Pick random site k=(i,j)
        Site x = randsite();
        Spin s = spin(x);
        flip(x);
        flipped.insert(x);
        addNeighbors(x,todo);
        
        while( !todo.empty() )
        {
            TodoItem t = todo.back();
            todo.pop_back();
            
            if( spin(t.x) != s || flipped.count(t.x) ) continue;
            
            if( rng_() > exp2j_[t.dir] )
            {
                flip(t.x);
                flipped.insert(t.x);
                addNeighbors(t.x,todo);
            }
        }
    }
    
    void measure()
    {
        int E = 0; // energy
        int M = 0; // magnetization
        for(size_t i = 0; i < L_[0]; ++i)
        {
            for(size_t j = 0; j < L_[1]; ++j)
            {
                E -= spins_[i][j]*(spins_[(i+1)%L_[0]][j] + spins_[i][(j+1)%L_[1]]);
                M += spins_[i][j];
            }
        }
        
        // Add sample to observables
        energy_ << E/double(nsites());
        double m = M/double(nsites());
        magnetization_ << m;
        abs_magnetization_ << fabs(m);
        m2_ << m*m;
        m4_ << m*m*m*m;
    }
    
    void save(std::string const & filename){
        alps::hdf5::archive ar(filename,alps::hdf5::archive::REPLACE);
        ar << alps::make_pvp("/parameters/L", L_[1]);
        ar << alps::make_pvp("/parameters/M", L_[0]);
        ar << alps::make_pvp("/parameters/Jx", J_[1]);
        ar << alps::make_pvp("/parameters/Jy", J_[0]);
        ar << alps::make_pvp("/parameters/SWEEPS", sweeps_);
        ar << alps::make_pvp("/parameters/THERMALIZATION", thermalization_);

        ar << alps::make_pvp("/simulation/results/"+energy_.representation(), energy_);
        ar << alps::make_pvp("/simulation/results/"+magnetization_.representation(), magnetization_);
        ar << alps::make_pvp("/simulation/results/"+abs_magnetization_.representation(), abs_magnetization_);
        ar << alps::make_pvp("/simulation/results/"+m2_.representation(), m2_);
        ar << alps::make_pvp("/simulation/results/"+m4_.representation(), m4_);

        alps::RealObsevaluator m = abs_magnetization_;
        alps::RealObsevaluator m2 = m2_;
        alps::RealObsevaluator m4 = m4_;

        alps::RealObsevaluator u2("Binder Cumulant U2");
        u2 = m2/(m*m);
        ar << alps::make_pvp("/simulation/results/"+u2.name(), u2);

        alps::RealObsevaluator chi("Connected Susceptibility");
        chi=nsites()*(m2-m*m);
        ar << alps::make_pvp("/simulation/results/"+chi.name(), chi);

        alps::RealObsevaluator u4("Binder Cumulant");
        u4 = m4/(m2*m2);
        ar << alps::make_pvp("/simulation/results/"+u4.name(), u4);
    }
    
protected:
    typedef signed char Spin;
    typedef std::pair<unsigned,unsigned> Site;
    typedef unsigned Direction;
    struct TodoItem { Site x; Direction dir; TodoItem(unsigned i=0, unsigned j=0, Direction d=0) : x(i,j), dir(d) {} };

    // Random int from the interval [0,max)
    int randint(int max) const
    {
        return static_cast<int>(max * rng_());
    }
    Site randsite() const
    {
        return Site(randint(L_[0]),randint(L_[1]));
    }
    size_t nsites() const
    {
        return L_[0]*L_[1];
    }
    Spin& spin(Site x)
    {
        return spins_[x.first][x.second];
    }
    const Spin& spin(Site x) const
    {
        return spins_[x.first][x.second];
    }
    void flip(Site x)
    {
        spin(x) *= -1;
    }
    void addNeighbors(Site x, vector<TodoItem>& n) const
    {
        using std::make_pair;
        n.push_back(TodoItem(x.first,(x.second+L_[1]-1)%L_[1],1)); // left
        n.push_back(TodoItem(x.first,(x.second      +1)%L_[1],1)); // right
        n.push_back(TodoItem((x.first+L_[0]-1)%L_[0],x.second,0)); // down
        n.push_back(TodoItem((x.first      +1)%L_[0],x.second,0)); // up
    }
    
    void printSpins() const
    {
        for( unsigned i = 0; i < L_[0]; ++i )
        {
            for( unsigned j = 0; j < L_[1]; ++j )
                cout << (int(spins_[i][j]) > 0 ? 'x' : '.');
            cout << "\n";
        }
        cout << endl;
    }

private:
    typedef boost::mt19937 engine_type;
    typedef boost::uniform_real<> distribution_type;
    typedef boost::variate_generator<engine_type&, distribution_type> rng_type;
    engine_type eng_;
    distribution_type dist_;
    mutable rng_type rng_;

    static const unsigned DIMENSIONS = 2;
    size_t L_[DIMENSIONS];
    double J_[DIMENSIONS];
    size_t sweeps_;
    size_t thermalization_;

    boost::multi_array<Spin,DIMENSIONS> spins_;
    double exp2j_[DIMENSIONS];

    alps::RealObservable energy_;
    alps::RealObservable magnetization_;
    alps::RealObservable abs_magnetization_;
    alps::RealObservable m2_;
    alps::RealObservable m4_;
    
    std::string filename_;
};

string usage(const std::string& prog)
{
    return "usage: " + prog + " M L Jy Jx N";
}
int main(int argc, const char** argv)
{
    if( argc != 6 )
    {
        cerr << usage(argv[0]) << endl;
        return -1;
    }
    
    size_t M, L, N;
    double Jy, Jx;
    try
    {
        using boost::lexical_cast;
        M   = lexical_cast<size_t>(argv[1]);    // Linear lattice sizes
        L   = lexical_cast<size_t>(argv[2]);    // 
        Jy  = lexical_cast<double>(argv[3]);    // coupling constants
        Jx  = lexical_cast<double>(argv[4]);    // 
        N   = lexical_cast<size_t>(argv[5]);    // # of simulation steps
    }
    catch(...)
    {
        cerr << usage(argv[0]) << endl;
        return -1;
    }
    

    std::cout << "# M: " << M  << "L: " << L << " Jy: " << Jy << " Jx: " << Jx << " N: " << N << std::endl;

    std::stringstream output_name;
    output_name << "ising_M" << M << "_L" << L << "_Jy" << Jy << "_Jx" << Jx <<".h5";
    Simulation sim(Jy, Jx, M, L, output_name.str());
    sim.run(N/5,N);

    return 0;
}
