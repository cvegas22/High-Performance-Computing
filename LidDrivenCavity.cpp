#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>

#define IDY(I,J) ((I)*Ny + (J))

#include "LidDrivenCavity.h"
#include "SolverCG.h"

// Constructor for the LidDrivenCavity class
LidDrivenCavity::LidDrivenCavity()
{
}

// Destructor for the LidDrivenCavity class
LidDrivenCavity::~LidDrivenCavity()
{
	delete cg;
	delete[] s;
	delete[] v;
	delete[] v_loc;
	delete[] tmp;
	delete[] tmp_loc;
}

/**
    @brief Function to set length of domain in xy directions.
    @param xlen Length of domain in x direction
    @param ylen Length of domain in y direction
    @return void
*/
void LidDrivenCavity::SetDomainSize(double xlen, double ylen)
{
    this->Lx = xlen;
    this->Ly = ylen;
}

/**
    @brief Function to set grid size of xy domain
    @param nx Number of grid points in x direction
    @param ny Number of grid points in y direction
    @return void
*/
void LidDrivenCavity::SetGridSize(int nx, int ny)
{
    this->Nx = nx;
    this->Ny = ny;
	// Total number of grid points
    Npts = Nx * Ny;
	
    // Grid spacing
    dx = Lx / (Nx-1);
    dy = Ly / (Ny-1);
}

/**
    @brief Function to set partition of xy domain for parallelisation
    @param part Number of partitions in the x and y directions
    @return void
*/
void LidDrivenCavity::SetPartition(int part)
{
    this->p = part;
}

/**
    @brief Function to set time step
    @param deltat Time step of each iteration
    @return void
*/
void LidDrivenCavity::SetTimeStep(double deltat)
{
    this->dt = deltat;
}

/**
    @brief Function to set final time
    @param finalt Final time
    @return void
*/
void LidDrivenCavity::SetFinalTime(double finalt)
{
    this->T = finalt;
}

/**
    @brief Function to set Reynolds number
    @param re Reynolds number
    @return void
*/
void LidDrivenCavity::SetReynoldsNumber(double re)
{
    this->Re = re;
    this->nu = 1.0/re;
}

/**
    @brief Function to perform domain partition and send the respective sub-domain information to each MPI process
    @param 
    @return void
*/
void LidDrivenCavity::SetMPI()
{
    // Check if MPI initalised
    MPI_Initialized(&MPIInit);
    if (!MPIInit){
        cout << "An error ocurred initialising MPI" << endl;
    } 
    else {
        // Get process rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Check if parallel
        if (size > 1) {
            if (rank == 0) {
                // Calculate number of elements for each partition (x-direction)
                int part_elem_x = (int) floor((double)Nx / (double)p);
                int part_remain_x = (int) Nx%p;

                // Calculate number of elements for each partition (y-direction)
                int part_elem_y = (int) floor((double)Ny / (double)p);
                int part_remain_y = (int) Ny%p;
                
                // Calculating start and end points of partitions (x-direction)
                int* part_start_x = new int [p];
                int* part_end_x = new int [p];
                int location_x = -1;
                for (int i=0; i<p; i++) {
                    part_start_x[i] = location_x+1;
                    location_x += part_elem_x;
                    if (part_remain_x>0) {
                        location_x++;
                        part_remain_x--;
                    }
                    part_end_x[i] = location_x;
                }

                // Calculating start and end points of partitions (y-direction)
                int* part_start_y = new int [p];
                int* part_end_y = new int [p];
                int location_y = -1;
                for (int i=0; i<p; i++) {
                    part_start_y[i] = location_y+1;
                    location_y += part_elem_y;
                    if (part_remain_y>0) {
                        location_y++;
                        part_remain_y--;
                    }
                    part_end_y[i] = location_y;
                }

                // Calculating start and end points of each sub-domain (x-y-direction)
                int* process_start_x = new int [size];
                int* process_end_x = new int [size];
                int* process_start_y = new int [size];
                int* process_end_y = new int [size];

                for (int j=0; j<p; j++) {
                    for (int i=0; i<p; i++) {
                        process_start_x[j*p+i] = part_start_x[i];
                        process_end_x[j*p+i] = part_end_x[i];
                        process_start_y[j*p+i] = part_start_y[j];
                        process_end_y[j*p+i] = part_end_y[j];
                    }
                }
				// Store first sub-domain positions at rank 0
				x_start = process_start_x[0];
				x_end = process_end_x[0];
				y_start = process_start_y[0];
				y_end = process_end_y[0];
                
                // Sending sub-domain positions to the different processes
                for (int dest=1; dest<size; dest++) {
                    MPI_Send(&process_start_x[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_end_x[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_start_y[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_end_y[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                }
                
                // Deallocating the temporary arrays
                delete[] process_start_x;
                delete[] process_start_y;
                delete[] process_end_x;
                delete[] process_end_y;
                delete[] part_start_x;
                delete[] part_start_y;
                delete[] part_end_x;
                delete[] part_end_y;
			}
			else {
				// Recieving sub-domain positions on each process 
                MPI_Recv(&x_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&x_end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&y_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				MPI_Recv(&y_end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
}

/**
    @brief Function that displays the chosen parameter values 
    @param
    @return void
*/
void LidDrivenCavity::PrintConfiguration()
{
	if(rank==0) {
		cout << "Grid size: " << Nx << " x " << Ny << endl;
		cout << "Spacing:   " << dx << " x " << dy << endl;
		cout << "Length:    " << Lx << " x " << Ly << endl;
		cout << "Grid pts:  " << Npts << endl;
		cout << "Timestep:  " << dt << endl;
		cout << "Steps:     " << ceil(T/dt) << endl;
		cout << "Reynolds number: " << Re << endl;
		cout << "Partitions in x and y directions: " << p << endl;
		cout << "Linear solver: preconditioned conjugate gradient" << endl;
		cout << endl;
		// Checks if chosen timestep meets the stability restriction
		if (nu * dt / dx / dy > 0.25) {
			cout << "ERROR: Time-step restriction not satisfied!" << endl;
			cout << "Maximum time-step is " << 0.25 * dx * dy / nu << endl;
			exit(-1);
		}
	}
}

void LidDrivenCavity::WriteSolution(std::string file)
{
	if(rank==0) {
		double* u0 = new double[Nx*Ny]();
		double* u1 = new double[Nx*Ny]();
		for (int i = 1; i < Nx - 1; ++i) {
			for (int j = 1; j < Ny - 1; ++j) {
				u0[IDY(i,j)] =  (s[IDY(i,j+1)] - s[IDY(i,j)]) / dy;
				u1[IDY(i,j)] = -(s[IDY(i+1,j)] - s[IDY(i,j)]) / dx;
			}
		}
		for (int i = 0; i < Nx; ++i) {
			u0[IDY(i,Ny-1)] = U;
		}

		std::ofstream f(file.c_str());
		std::cout << "Writing file " << file << std::endl;
		int k = 0;
		for (int i = 0; i < Nx; ++i) {
			for (int j = 0; j < Ny; ++j) {
				k = IDY(i, j);
				f << i*dx << " " << j*dy << " " << v[k] <<  " " << s[k] << " " << u0[k] << " " << u1[k] << std::endl;
			}
			f << std::endl;
		}
		f.close();

		delete[] u0;
		delete[] u1;
	}
}

/**
    @brief Function that solves the vorticity and streamfunction at each timestep from t=0 to t=T
    @param
    @return void
*/
void LidDrivenCavity::Integrate()
{	
    double dxi  = 1.0/dx;
    double dyi  = 1.0/dy;
    double dx2i = 1.0/dx/dx;
    double dy2i = 1.0/dy/dy;
	double t = 0.0;
	
	// Initialise streamfunction array to zeros
	s = new double[Npts]();
	// Initialise constructor of SolverCG
	cg  = new SolverCG(Nx, Ny, dx, dy);
	
	double time1, time2,duration,global;
	time1 = MPI_Wtime();
	
	while (t < T){
		// Initialise vorticity arrays to zeros at each time-step
		v = new double[Npts]();
		v_loc = new double[Npts]();
		tmp = new double[Npts]();
		tmp_loc = new double[Npts]();
		
		if(rank==0) {
			std::cout << "Step: " << setw(8) << t/dt << "  Time: " << setw(8) << t << std::endl;
		}

		// Vorticity boundary conditions (serial/OpenMP)
		if(size == 1) {
			for (int i = 1; i < Nx-1; ++i) {
				// bottom
				v[IDY(i,0)]    = 2.0 * dy2i * (s[IDY(i,0)]    - s[IDY(i,1)]);
				// top
				v[IDY(i,Ny-1)] = 2.0 * dy2i * (s[IDY(i,Ny-1)] - s[IDY(i,Ny-2)]) - 2.0 * dyi*U;
			}
			for (int j = 1; j < Ny-1; ++j) {
				// left
				v[IDY(0,j)]    = 2.0 * dx2i * (s[IDY(0,j)]    - s[IDY(1,j)]);
				// right
				v[IDY(Nx-1,j)] = 2.0 * dx2i * (s[IDY(Nx-1,j)] - s[IDY(Nx-2,j)]);
			}
		}
		

		if (size > 1) {
			// Compute interior vorticity and boundary conditions (MPI)
			for (int i = x_start; i < 1+x_end; ++i) {
				for (int j = y_start; j < 1+y_end; ++j) {
					// bottom
					if (j == 0) {
						v_loc[IDY(i,0)]    = 2.0 * dy2i * (s[IDY(i,0)]    - s[IDY(i,1)]);
					} 
					// top
					else if (j == Ny-1) {
						v_loc[IDY(i,Ny-1)] = 2.0 * dy2i * (s[IDY(i,Ny-1)] - s[IDY(i,Ny-2)]) - 2.0 * dyi*U;
					}
					// left
					else if (i == 0) {
						v_loc[IDY(0,j)]    = 2.0 * dx2i * (s[IDY(0,j)]    - s[IDY(1,j)]);
					}
					// right
					else if (i == Nx-1) {
						v_loc[IDY(Nx-1,j)] = 2.0 * dx2i * (s[IDY(Nx-1,j)] - s[IDY(Nx-2,j)]);
					}
					// interior vorticity
					else {
						v_loc[IDY(i,j)] = dx2i*(2.0 * s[IDY(i,j)] - s[IDY(i+1,j)] - s[IDY(i-1,j)])
								+ 1.0/dy/dy*(2.0 * s[IDY(i,j)] - s[IDY(i,j+1)] - s[IDY(i,j-1)]);
					}
				}
			}
			// BC at the edges
			v_loc[IDY(0,0)] = 0.0;
			v_loc[IDY(0,Ny-1)] = 0.0;
			v_loc[IDY(Nx-1,0)] = 0.0;
			v_loc[IDY(Nx-1,Ny-1)] = 0.0;
			// Combine local vorticity arrays into full domain and send to each process
			MPI_Allreduce(v_loc, v, Npts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		}
		else {
			// Compute interior vorticity (serial/OpenMP)
			for (int i = 1; i < Nx - 1; ++i) {
			for (int j = 1; j < Ny - 1; ++j) {
					v[IDY(i,j)] = dx2i*(2.0 * s[IDY(i,j)] - s[IDY(i+1,j)] - s[IDY(i-1,j)])
							+ 1.0/dy/dy*(2.0 * s[IDY(i,j)] - s[IDY(i,j+1)] - s[IDY(i,j-1)]);
				}
			}
		}

		if (size > 1) {
			// Time advance vorticity (MPI)
			for (int i = x_start; i < 1+x_end; ++i) {
				for (int j = y_start; j < 1+y_end; ++j) {
					if ((i != 0) & (i != Nx-1) & (j != 0) & (j != Ny-1)) {
						tmp_loc[IDY(i,j)] = v[IDY(i,j)] + dt*(( (s[IDY(i+1,j)] - s[IDY(i-1,j)]) * 0.5 * dxi
											*(v[IDY(i,j+1)] - v[IDY(i,j-1)]) * 0.5 * dyi)
											- ( (s[IDY(i,j+1)] - s[IDY(i,j-1)]) * 0.5 * dyi
											*(v[IDY(i+1,j)] - v[IDY(i-1,j)]) * 0.5 * dxi)
											+ nu * (v[IDY(i+1,j)] - 2.0 * v[IDY(i,j)] + v[IDY(i-1,j)])*dx2i
											+ nu * (v[IDY(i,j+1)] - 2.0 * v[IDY(i,j)] + v[IDY(i,j-1)])*dy2i);
					} else {
						tmp_loc[IDY(i,j)] = v[IDY(i,j)];
					}
				}
			}
			// Combine updated local vorticity arrays to full domain and send to each process
			MPI_Allreduce(tmp_loc, tmp, Npts, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		}
		else {
			// Time advance vorticity (serial/OpenMP)
			for (int i = 1; i < Nx - 1; ++i) {
				for (int j = 1; j < Ny - 1; ++j) {
					tmp[IDY(i,j)] = v[IDY(i,j)] + dt*(( (s[IDY(i+1,j)] - s[IDY(i-1,j)]) * 0.5 * dxi
						*(v[IDY(i,j+1)] - v[IDY(i,j-1)]) * 0.5 * dyi)
					- ( (s[IDY(i,j+1)] - s[IDY(i,j-1)]) * 0.5 * dyi
						*(v[IDY(i+1,j)] - v[IDY(i-1,j)]) * 0.5 * dxi)
					+ nu * (v[IDY(i+1,j)] - 2.0 * v[IDY(i,j)] + v[IDY(i-1,j)])*dx2i
					+ nu * (v[IDY(i,j+1)] - 2.0 * v[IDY(i,j)] + v[IDY(i,j-1)])*dy2i);
				}
			}
			for (int i = 0; i < Nx; ++i) {
				tmp[IDY(i,0)] = v[IDY(i,0)];
				tmp[IDY(i,Ny-1)] = v[IDY(i,Ny-1)];
			}
			for (int j = 0; j < Ny; ++j) {
				tmp[IDY(0,j)] = v[IDY(0,j)];
				tmp[IDY(Nx-1,j)] = v[IDY(Nx-1,j)];
			}
		}

		// Create domain partitions in SolverCG
		if (t == 0) {
			cg->SetMPI(Nx,Ny,p);
		}
		
		// Solve Poisson problem
		cg->Solve(tmp, s);
		
		// Obtain updated streamfunction from SolverCG and share to all other processes (MPI)
		if (size > 1) {
			if (rank == 0) {
				cg->UpdateStreamfunction(s);
			}
			MPI_Bcast(s, Npts, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		}

		t += dt;
		
		// dealocate memory at each timestep
		delete[] v;
		delete[] v_loc;
		delete[] tmp;
		delete[] tmp_loc;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	time2 = MPI_Wtime();
	duration = time2 - time1;
	MPI_Reduce(&duration,&global,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
	if(rank == 0) {
		printf("Global runtime is %f\n",global);
	}
}
