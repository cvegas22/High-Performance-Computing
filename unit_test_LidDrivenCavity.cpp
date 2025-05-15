#define BOOST_TEST_MODULE LidDrivenCavity
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <cmath>
#include "LidDrivenCavity.h" 
using namespace std;

// Function to compute index in 1D array from 2D domain
#define IDY(I,J) ((I)*Ny + (J))

// BOOST unit test
BOOST_AUTO_TEST_CASE(TestLid) {
    double dt = 0.01;
    double T = 0.1;
    int    Nx = 9;
    int    Ny = 9;
    int    p = 1;
    double Lx = 1.0;
    double Ly = 1.0;
    double Re = 10;
	
	int n = Nx*Ny;
    double dx = 1.0 / (Nx - 1);
    double dy = 1.0 / (Ny - 1);
	
	double* v = new double[n]();
	double* vAnal = new double[n]();
	
    // Initialize the LidDrivenCavity instance
    LidDrivenCavity* solver = new LidDrivenCavity();

    // Setting up and running solver
    solver->SetDomainSize(Lx,Ly);
    solver->SetGridSize(Nx,Ny);
	solver->SetPartition(p);
    solver->SetTimeStep(dt);
    solver->SetFinalTime(T);
    solver->SetReynoldsNumber(Re);
    solver->Integrate();
	
	// Compute analytical solution
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
            vAnal[IDY(i,j)] = -M_PI * M_PI * ((k * k + l * l)
                            * sin(M_PI * k * i * dx)
                            * sin(M_PI * l * j * dy));
        }
    }

    // Check accuracy of each array entry
	for (int i = 0; i < n; ++i) {
		BOOST_CHECK_CLOSE(v[i],vAnal[i],1e-5);
	}
}