#include <iostream>
using namespace std;
#include <mpi.h>
#include <omp.h>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

#include "LidDrivenCavity.h"

int main(int argc, char **argv)
{
    // Setting up command line options
    po::options_description opts(
        "Solver for the 2D lid-driven cavity incompressible flow problem");
    opts.add_options()
        ("Lx",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the x-direction.")
        ("Ly",  po::value<double>()->default_value(1.0),
                 "Length of the domain in the y-direction.")
        ("Nx",  po::value<int>()->default_value(9),
                 "Number of grid points in x-direction.")
        ("Ny",  po::value<int>()->default_value(9),
                 "Number of grid points in y-direction.")
        ("p",   po::value<int>()->default_value(1),
                 "Number of partitions in the x and y directions.")
        ("dt",  po::value<double>()->default_value(0.01),
                 "Time step size.")
        ("T",   po::value<double>()->default_value(0.1),
                 "Final time.")
        ("Re",  po::value<double>()->default_value(10),
                 "Reynolds number.")
        ("verbose",    "Be more verbose.")
        ("help",       "Print help message.");

    // Use boost to parse command-line arguments using list of possible options and generate map of options
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opts), vm);
    po::notify(vm);

    // Prints solver usage if user used the "--help" option
    if (vm.count("help")) {
        cout << opts << endl;
        return 0;
    }

    // Extracting options and saving into variables
    const double Lx = vm["Lx"].as<double>();
    const double Ly = vm["Ly"].as<double>();
    const int    Nx = vm["Nx"].as<int>();
    const int    Ny = vm["Ny"].as<int>();
    const int    p = vm["p"].as<int>();
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const double Re = vm["Re"].as<double>();

    // Initialise MPI.
    int retval = MPI_Init(&argc, &argv);
    if (retval != MPI_SUCCESS) {
        cout << "An error ocurred initialising MPI" << endl;
        return 0;
    }

    int rank, size, retval_rank, retval_size;
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &rank); // zero-based
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM) {
        cout << "Invalid communicator" << endl;
        return 0;
    }

    // Check number of processes is suitable
    if ((size != p*p) || (p < 0)) {
        if (rank == 0) {
            cout << "Parallelisation requires P = p*p MPI ranks for integer p > 0" << endl;
            MPI_Finalize();
        }

        return 0;
    }

    // New instance of the LidDrivenCavity class
    LidDrivenCavity* solver = new LidDrivenCavity();

    // Setting up solver
    solver->SetDomainSize(Lx,Ly);
    solver->SetGridSize(Nx,Ny);
	solver->SetPartition(p);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
	solver->SetMPI();
    solver->PrintConfiguration();

    // Running solver
    //solver->Initialise();
    solver->Integrate();
    solver->WriteSolution("final.txt");
	


    // Finalise MPI.
    MPI_Finalize();

	return 0;
}
