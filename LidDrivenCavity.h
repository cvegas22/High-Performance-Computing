//#pragma once

#include <string>
using namespace std;

class SolverCG;

class LidDrivenCavity
{
public:
    // Constructor and Destructor
    LidDrivenCavity();
    ~LidDrivenCavity();
	
    void SetDomainSize(double xlen, double ylen);
    void SetGridSize(int nx, int ny);
    void SetPartition(int part);
    void SetTimeStep(double deltat);
    void SetFinalTime(double finalt);
    void SetReynoldsNumber(double Re);
    void SetMPI();

    void Integrate();
    void WriteSolution(std::string file);
    void PrintConfiguration();

private:
    double* v   = nullptr;
    double* s   = nullptr;
	double* tmp   = nullptr;
	double* tmp_loc   = nullptr;
	double* v_loc   = nullptr;
	int x_start;
	int x_end;
	int y_start;
	int y_end;

    double dt;
    double T;
    double dx;
    double dy;
    int    Nx;
    int    Ny;
    int    Npts;
    int    p;
    double Lx;
    double Ly;
    double Re;
    double U = 1.0;
    double nu;

    int MPIInit;
    int rank;
    int size;

    SolverCG* cg = nullptr;
};

