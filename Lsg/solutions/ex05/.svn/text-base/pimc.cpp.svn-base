#include <iostream>
#include <alps/ngs.hpp>

class PimcSimulation : public alps::mcbase {
    public:
        PimcSimulation(parameters_type const & params, std::size_t seed_offset = 0)
            : alps::mcbase(params, seed_offset)
            , sweeps_(0)
            , thermalization_sweeps_(int(params["THERMALIZATION"]))
            , total_sweeps_(int(params["SWEEPS"]))
            , m_(params["M"])
            , max_displacement_(params["MAXDISPLACEMENT"])
            , beta_(1. / double(params["T"]))
            , omega_(params["OMEGA"])
            , tau_(beta_/m_)
            , config_(m_)
        {
            for( size_t i = 0; i < m_; ++i )
                config_[i] = random() - 0.5;
            results.create_RealObservable("Energy");
            results.create_RealObservable("PotentialEnergy");
            results.create_RealObservable("KineticEnergy");
            results.create_RealObservable("AcceptanceRate");
        }
        void do_update() 
        {
            ++sweeps_;
            int acc = 0;
            for( size_t i = 0; i < m_; ++i ) 
                acc += do_step();
            results["AcceptanceRate"] << acc/double(m_);
        };
        bool do_step()
        {
            using std::exp;
            size_t j = static_cast<size_t>(m_ * random());
            double oldx = config_[j];
            double newx = oldx + (2.*random()-1.)*max_displacement_;
            double xprev = config_[(j-1+m_)%m_];
            double xnext = config_[(j+1   )%m_];
            double tdiff = ( (newx-xprev)*(newx-xprev) + (newx-xnext)*(newx-xnext)
                           - (oldx-xprev)*(oldx-xprev) - (oldx-xnext)*(oldx-xnext) 
                            ) / (2.*tau_);
            double vdiff = tau_*.5*omega_ * (newx*newx - oldx*oldx);
            double logchi = -tdiff - vdiff;
            if( logchi > 0. || random() < exp(logchi) )
            {
                config_[j] = newx;
                return true;
            }
            else
                return false;
        }
        void do_measurements() 
        {
            if( sweeps_ <= thermalization_sweeps_ )
                return;

            double v = potential_energy();
            double t = kinetic_energy();
            results["Energy"] << v+t;
            results["PotentialEnergy"] << v;
            results["KineticEnergy"] << t;
        };
        double fraction_completed() const {
            return (sweeps_ < thermalization_sweeps_ ? 0. : ( sweeps_ - thermalization_sweeps_ ) / double(total_sweeps_));
        }
        double potential_energy() const
        {
            double v = 0.;
            for( size_t i = 0; i < m_; ++i )
                v += config_[i]*config_[i];
            return .5*omega_*omega_/m_*v;
        }
        double kinetic_energy() const
        {
            double t = (config_.front()-config_.back())*(config_.front()-config_.back());
            for( size_t i = 1; i < m_; ++i )
                t += (config_[i]-config_[i-1])*(config_[i]-config_[i-1]);
            return .5/tau_ - .5*t/(m_*tau_*tau_);
        }
        double analytic_energy() const
        {
            return .5*omega_/std::tanh(.5*beta_*omega_);
        }
        
    private:
        int sweeps_;
        int thermalization_sweeps_;
        int total_sweeps_;
        size_t m_;
        double max_displacement_;
        double beta_;
        double omega_;
        double tau_;
        std::vector<double> config_;
};

void print_results(std::ostream& os,const alps::mcresults& results)
{
    for(alps::mcresults::const_iterator it = results.begin(); it != results.end(); ++it)
        os << std::fixed << std::setprecision(5) << it->first << ":\t" << it->second.mean<double>() << "\t+- " << it->second.error<double>() << "\ttau = " << it->second.tau<double>() << std::endl;
}

bool stop_callback(boost::posix_time::ptime const & end_time) {
    static alps::mcsignal signal;
    return !signal.empty() || boost::posix_time::second_clock::local_time() > end_time;
}
int main(int argc, char *argv[]) {
    alps::mcoptions options(argc, argv);
    if (options.valid && options.type == alps::mcoptions::SINGLE) {
        alps::parameters_type<PimcSimulation>::type params(options.input_file);
        std::string inbase = options.input_file.substr(0,options.input_file.find_last_of('.'));
        std::string dumpfile = inbase+".run";
        PimcSimulation s(params);
        if (options.resume)
            s.load(dumpfile);
//        s.run(boost::bind(&stop_callback, boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(options.time_limit)));

        while( s.fraction_completed() < 1. )
        {
            s.do_update();
            s.do_measurements();
        }

        s.save(dumpfile);
        alps::results_type<PimcSimulation>::type results = collect_results(s);
        std::cout << "Stopped after " << results["Energy"].count() << " sweeps.\n";
        print_results(std::cout,results);
        std::cout << "Analytic value: E=" << s.analytic_energy() << std::endl;
        save_results(results, params, options.output_file, "/simulation/results");
    } 
    else if(options.valid && options.type == alps::mcoptions::MPI) 
    {
        boost::mpi::environment env(argc, argv);
        boost::mpi::communicator c;
        std::string inbase = options.input_file.substr(0,options.input_file.find_last_of('.'));
        std::string dumpfile = inbase+".run"+boost::lexical_cast<std::string>(c.rank());
        alps::parameters_type<PimcSimulation>::type params(options.input_file);
        alps::mcmpisim<PimcSimulation> s(params, c);
        if (options.resume)
            s.load(dumpfile);
        s.run(boost::bind(&stop_callback, boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(options.time_limit)));
        s.save(dumpfile);
        alps::results_type<alps::mcmpisim<PimcSimulation> >::type results = collect_results(s);
        if (!c.rank()) {
            save_results(results, params, options.output_file, "/simulation/results");
            std::cout << results;
        }
    }
}
