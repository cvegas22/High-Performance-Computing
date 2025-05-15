#define BOOST_TEST_MODULE SolverCG
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <cmath>
#include "SolverCG.h" 
using namespace std;

// Function to compute index in 1D array from 2D domain
#define IDY(I,J) ((I)*Ny + (J))

// BOOST unit test
BOOST_AUTO_TEST_CASE(TestSolverCG) {
    int Nx = 100; // Grid size in x
    int Ny = 100; // Grid size in y
	int n = Nx*Ny;
    double dx = 1.0 / (Nx - 1);
    double dy = 1.0 / (Ny - 1);
	
	double* v = new double[n]();
	double* s = new double[n]();
	double* sAnal = new double[n]();
	
    // Initialize the SolverCG instance
    SolverCG* cg = new SolverCG(Nx, Ny, dx, dy);

    // Solve the system
    cg->Solve(v, s);
	
	// Compute analytical solution
    const int k = 3;
    const int l = 3;
    for (int i = 0; i < Nx; ++i) {
        for (int j = 0; j < Ny; ++j) {
			sAnal[IDY(i,j)] = -sin(M_PI*k*i*dx)*sin(M_PI*l*j*dy);
        }
    }

    // Check accuracy of each array entry
	for (int i = 0; i < n; ++i) {
		BOOST_CHECK_CLOSE(s[i],sAnal[i],1e-5);
	}
}
