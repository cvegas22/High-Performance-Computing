//#pragma once

class SolverCG
{
public:
    SolverCG(int pNx, int pNy, double pdx, double pdy);
    ~SolverCG();

    void Solve(double* v, double* s);
	void SetMPI(int Nx, int Ny, int part);
	void UpdateStreamfunction(double* stream);

private:
    double dx;
    double dy;
    int Nx;
    int Ny;
	int part;
	int n;
	int n_layered;
	double* r = nullptr;
    double* b = nullptr;
    double* x = nullptr;
	double* x_full = nullptr;
	double* x_sum = nullptr;
	double* x_layered = nullptr;
    double* p = nullptr;
	double* p_layered = nullptr;
    double* z = nullptr;
    double* t = nullptr;
	double* v;
	double* s;
	int x_start;
	int x_end;
	int y_start;
	int y_end;
	int ArraySize;
	int height;
	int width;
	int i_start;
	int i_end;
	int j_start;
	int j_end;
	int pos_x;
	int pos_y;
    double alpha;
	double alpha1;
	double alpha1_sum;
	double alpha2;
    double beta;
	double beta1;
	double beta1_sum;
	double beta2;
    double eps;
	double eps_squared;
	double eps_squared_sum;
	
	int MPIInit;
    int rank;
    int size;

    void ApplyOperator(double* in, double* out);
    void Precondition(double* in, double* out);
    void ImposeBC(double* inout);
	void CreatePartitions(double* in1, double* in2, double* out1, double* out2);
	void PartitionToDomain(double* x, double* x_full);
	void AddLayer(double* p, double* p_layered);
	double DotProd(double* in1, double* in2);
	void Copy(double* in, double* out);
	void Daxpy(double* in, double* out, double c);

};

