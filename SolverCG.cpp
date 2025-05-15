#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cmath>
using namespace std;

#include <cblas.h>
#include <mpi.h>
#include <omp.h>
#include <thread>
#include "SolverCG.h"

#define IDY(I,J) ((I)*Ny + (J))
#define IDY_loc(I,J) ((I-x_start)*height + (J-y_start))
#define IDY_loc_layered(I,J) (I+1-x_start)*(height+2) + (J+1-y_start)

// Constructor for SolverCG class
SolverCG::SolverCG(int pNx, int pNy, double pdx, double pdy)
{
    dx = pdx;
    dy = pdy;
    Nx = pNx;
    Ny = pNy;
    v = new double[Nx*Ny]();
    s = new double[Nx*Ny]();
}

// Destructor for SolverCG class
SolverCG::~SolverCG()
{
    // Dealocates memory of class arrays
    delete[] b;
    delete[] x;
    delete[] r;
    delete[] x_full;
    delete[] p;
    delete[] p_layered;
    delete[] x_sum;
    delete[] z;
    delete[] t;
    delete[] v;
    delete[] s;
}

/**
    @brief Function to perform domain partition and send the respective sub-domain information to each MPI process
    @param Nx Number of grid-points in x-direction
    @param Ny Number of grid-points in y-direction
    @param p Number of partition in the x and y directions
    @return void
*/
void SolverCG::SetMPI(int Nx, int Ny, int p){
    part = p;
    // Check if MPI initalised
    MPI_Initialized(&MPIInit);
    if (!MPIInit){
        cout << "An error ocurred initialising MPI" << endl;
    } 
    else {
        // Get process rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // Check if serial or parallel
        if (size > 1) {
            if (rank == 0) {
                // Calculate number of elements for each partition (x-direction)
                int part_elem_x = (int) floor((double)Nx / (double)part);
                int part_remain_x = (int) Nx%part;

                // Calculate number of elements for each partition (y-direction)
                int part_elem_y = (int) floor((double)Ny / (double)part);
                int part_remain_y = (int) Ny%part;
                
                // Calculating start and end points of partitions (x-direction)
                int* part_start_x = new int [part];
                int* part_end_x = new int [part];
                int location_x = -1;
                for (int i=0; i<part; i++) {
                    part_start_x[i] = location_x+1;
                    location_x += part_elem_x;
                    if (part_remain_x>0) {
                        location_x++;
                        part_remain_x--;
                    }
                    part_end_x[i] = location_x;
                }

                // Calculating start and end points of partitions (y-direction)
                int* part_start_y = new int [part];
                int* part_end_y = new int [part];
                int location_y = -1;
                for (int i=0; i<part; i++) {
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
		int* length = new int [size];
		int* count_x = new int [size];
		int* count_y = new int [size];

                for (int j=0; j<part; j++) {
                    for (int i=0; i<part; i++) {
                        process_start_x[j*part+i] = part_start_x[i];
                        process_end_x[j*part+i] = part_end_x[i];
                        process_start_y[j*part+i] = part_start_y[j];
                        process_end_y[j*part+i] = part_end_y[j];
			length[j*part+i] = (1 + part_end_x[i] - part_start_x[i])*(1 + part_end_y[j] - part_start_y[j]);
			count_x[j*part+i] = i;
			count_y[j*part+i] = j;
                    }
                }
		// Store first sub-domain positions at rank 0
		x_start = process_start_x[0];
		x_end = process_end_x[0];
		y_start = process_start_y[0];
		y_end = process_end_y[0];
		ArraySize = length[0];
		pos_x = count_x[0];
		pos_y = count_y[0];
                
                // Sending sub-domain positions and information of loaction to the different processes
                for (int dest=1; dest<size; dest++) {
                    MPI_Send(&process_start_x[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_end_x[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_start_y[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
                    MPI_Send(&process_end_y[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		    MPI_Send(&length[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		    MPI_Send(&count_x[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
		    MPI_Send(&count_y[dest], 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
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
		delete[] length;
		delete[] count_x;
		delete[] count_y;
	    }
	    else {
	        // Recieving sub-domain positions and location information on each process 
                MPI_Recv(&x_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&x_end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&y_start, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&y_end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&ArraySize, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&pos_x, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&pos_y, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	    }
			
	    // Compute additional sub-domain information for SolverCG calculations
	    height = (1+y_end-y_start);
	    width = (1+x_end-x_start);
	    j_start = y_start;
	    i_start = x_start;
	    j_end = y_end;
	    i_end = x_end;
	    n_layered = (height+2)*(width+2);
	    // Provide new start and end positions if sub-domain is within domain boundary (to facilitate some calculations)
	    if (y_start == 0){ j_start = y_start+1;}
	    if (x_start == 0){ i_start = x_start+1;}
	    if (y_end == Ny-1){ j_end = y_end-1;}
	    if (x_end == Nx-1){ i_end = x_end-1;}
        }
    }
}

/**
    @brief Function that solves the Poisson problem and computes streamfunction at time t + dt
    @param v Vorticity array (full domain)
    @param s Streamfunction array (full domain)
    @return void
*/
void SolverCG::Solve(double* v, double* s) {
	int it;
	int n = Nx*Ny;
	int N = Nx*Ny;
	int threads = 0;
	double tol = 0.000001;
	
	#pragma omp parallel
	{
		threads = omp_get_num_threads();
	}
	
	// Update array size to sub-domain size (MPI)
	if (size > 1) {	
		n = ArraySize;
	} 
	
	// Initialise arrays to zeros
	b = new double[n]();
	x = new double[n]();
	r = new double[n]();
	p = new double[n]();
	z = new double[n]();
	t = new double[n]();
	x_full = new double[N]();
	x_sum = new double[N]();
	
	// check if convergence criteria is already met
	if (rank == 0) {
		eps = cblas_dnrm2(N, v, 1);
		if (eps < tol) {
			std::fill(v, v+N, 0.0);
			cout << "Norm is " << eps << endl;
			return;
		}
	}

        it = 0;
	if (size > 1) {
		// Obtain vorticity and sreamfunction sub-domain array of respective rank
		CreatePartitions(v, s, b, x);
		x_layered = new double[n_layered]();
		AddLayer(x,x_layered);
		
		ApplyOperator(x_layered, t);
		delete[] x_layered;
		
		cblas_dcopy(n, b, 1, r, 1);        // r_0 = b 
		ImposeBC(r);
		cblas_daxpy(n, -1.0, t, 1, r, 1);
		Precondition(r, z);
		cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0
		
		p_layered = new double[n_layered]();
		AddLayer(p,p_layered);
		
		do {
			it++;
			// Perform action of Nabla^2 * p
			ApplyOperator(p_layered, t);
			delete[] p_layered;
			
			alpha = 0.0;
			alpha1_sum = 0.0;
			alpha1 = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
			MPI_Allreduce(&alpha1, &alpha1_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			alpha2 = cblas_ddot(n, r, 1, z, 1) / alpha1_sum; // compute alpha_k
			MPI_Allreduce(&alpha2, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			beta1_sum = 0.0;
			beta1  = cblas_ddot(n, r, 1, z, 1);  // z_k^T r_k
			MPI_Allreduce(&beta1, &beta1_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
			cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k
		
			eps_squared_sum = 0.0;
			eps_squared = cblas_ddot(n, r, 1, r, 1);
			MPI_Allreduce(&eps_squared, &eps_squared_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			eps = eps_squared_sum;
			if (eps < tol*tol) {
				break;
			}
		
			Precondition(r, z);
        
			beta = 0.0;
			beta2 = cblas_ddot(n, r, 1, z, 1) / beta1_sum;
			MPI_Allreduce(&beta2, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

			cblas_dcopy(n, z, 1, t, 1);
			cblas_daxpy(n, beta, p, 1, t, 1);
			cblas_dcopy(n, t, 1, p, 1);
		
			p_layered = new double[n_layered]();
			AddLayer(p,p_layered);

		} while (it < 5000); // Set a maximum number of iterations
		
	} else if (threads > 1) {
		b = v;
		x = s;
		
	        ApplyOperator(x, t);
		Copy(b,r);
		ImposeBC(r);
		Daxpy(t,r,-1.0);
		Precondition(r, z);
		Copy(z,p);
		 do {
			it++;
			// Perform action of Nabla^2 * p
			ApplyOperator(p, t);
			
			alpha1 = DotProd(t,p);
			alpha = DotProd(r,z)/alpha1;
			beta1 = DotProd(r,z);

			Daxpy(p,x,alpha);
			Daxpy(t,r,-alpha);

			eps_squared = DotProd(r,r);
			eps = sqrt(eps_squared);
			if (eps < tol) {
				break;
			}
		
			Precondition(r, z);
			beta = DotProd(r,z)/beta1;
			Copy(z,t);
			Daxpy(p,t,beta);
			Copy(t,p);

		} while (it < 5000); // Set a maximum number of iterations
		
	} else {
		b = v;
		x = s;
		
	        ApplyOperator(x, t);
		cblas_dcopy(n, b, 1, r, 1);        // r_0 = b 
		ImposeBC(r);
		cblas_daxpy(n, -1.0, t, 1, r, 1);
		Precondition(r, z);
		cblas_dcopy(n, z, 1, p, 1);        // p_0 = r_0
		 do {
			it++;
			// Perform action of Nabla^2 * p
			ApplyOperator(p, t);
		
			alpha1 = cblas_ddot(n, t, 1, p, 1);  // alpha = p_k^T A p_k
			if (it == 1) {
				alpha2 = cblas_ddot(n, r, 1, z, 1); 
			} else{
				alpha2 = beta2;
			}
			
			alpha = alpha2/alpha1; // compute alpha_k
			cblas_daxpy(n,  alpha, p, 1, x, 1);  // x_{k+1} = x_k + alpha_k p_k
			cblas_daxpy(n, -alpha, t, 1, r, 1); // r_{k+1} = r_k - alpha_k A p_k

			// only compute eps every 4 iterations (doesn't affect convergence and reduces compute time for large domains)
			if (it%4 == 0) {
				eps = cblas_dnrm2(n, r, 1);
				if (eps < tol) {
					break;
				}
			}
		
			Precondition(r, z);
			
			beta1 = alpha2; // z_k^T r_k
			beta2 = cblas_ddot(n, r, 1, z, 1);
			beta = beta2 / beta1;
			cblas_dcopy(n, z, 1, t, 1);
			cblas_daxpy(n, beta, p, 1, t, 1);
			cblas_dcopy(n, t, 1, p, 1);

		} while (it < 5000); // Set a maximum number of iterations
	}
	
	if (it == 5000) {
		cout << "FAILED TO CONVERGE" << endl;
		exit(-1);
	}

	if (rank == 0) { 
		cout << "Converged in " << it << " iterations. eps = " << eps << endl;
	}
	
	if (size > 1) {
		PartitionToDomain(x, x_full);
		MPI_Allreduce(x_full, x_sum, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		delete[] x_full;
	} else {
		s = x;
	}
}

/**
    @brief Function to send updated streamfunction values to LidDrivenCavity
    @param stream Updated stream-function array (full-domain)
    @return void
*/
void SolverCG::UpdateStreamfunction(double* stream){
	for (int i = 0; i < Nx; ++i) {
		for (int j = 0; j < Ny; ++j) {
			stream[IDY(i,j)] = x_sum[IDY(i,j)];
		}
	}
}

/**
    @brief Function to create sub-domain arrays from full domain arrays
    @param in1,in2 Entering full domain arrays
    @param out1,out2 Exiting sub-domain arrays
    @return void
*/
void SolverCG::CreatePartitions(double* in1, double* in2, double* out1, double* out2) {
        int i, j;
	int k = 0;
	for (i = x_start; i < 1+x_end; ++i) {
		for (j = y_start; j < 1+y_end; ++j) {
			out1[k] = in1[IDY(i,j)];
			out2[k] = in2[IDY(i,j)];
			k++;
		}
	}
}

/**
    @brief Function to apply equation 4 to each interior value of domain
    @param in Entering sub-domain array (with additional outer layer)
    @param out Exiting array (with size equal to unlayered sub-domain array)
    @return void
*/
void SolverCG::ApplyOperator(double* in, double* out) {
        double dx2i = 1.0/dx/dx;
        double dy2i = 1.0/dy/dy;
	if (size > 1) {
		for (int i = i_start; i < 1+i_end; ++i) {
			for (int j = j_start; j < 1+j_end; ++j) {
				out[IDY_loc(i,j)] = ( -   in[IDY_loc_layered(i-1,j)]
									+ 2.0*in[IDY_loc_layered(i,j)]
									-     in[IDY_loc_layered(i+1,j)])*dx2i
								+ ( -     in[IDY_loc_layered(i,j-1)]
									+ 2.0*in[IDY_loc_layered(i,j)]
									-     in[IDY_loc_layered(i,j+1)])*dy2i;
			}
		}
	} else {
		#pragma omp parallel for collapse(2)
		for (int i = 1; i < Nx - 1; ++i) {
			for (int j = 1; j < Ny - 1; ++j) {
				out[IDY(i,j)] = ( -   in[IDY(i-1, j)]
								+ 2.0*in[IDY(i,   j)]
								-     in[IDY(i+1, j)])*dx2i
							+ ( -     in[IDY(i, j-1)]
								+ 2.0*in[IDY(i,   j)]
								-     in[IDY(i, j+1)])*dy2i;
			}
		}
	}
}

/**
    @brief Function to apply a scaling factor to interior values of domain
    @param in Entering array
    @param out Exiting array with scaling factor
    @return void
*/
void SolverCG::Precondition(double* in, double* out) {
	int k = 0;
        double dx2i = 1.0/dx/dx;
        double dy2i = 1.0/dy/dy;
        double factor = 2.0*(dx2i + dy2i);
	if (size > 1) {
		for (int i = x_start; i < 1+x_end; ++i) {
			for (int j = y_start; j < 1+y_end; ++j) {
				if ((j == 0) | (j == Ny-1) | (i == 0) | (i == Nx-1)) {
					out[k] = in[k];
				} else {
					out[k] = in[k]/factor;
				}
				k++;
			}
		}
	} else {
		#pragma omp parallel for collapse(2)
		for (int i = 1; i < Nx - 1; ++i) {
			for (int j = 1; j < Ny - 1; ++j) {
				out[IDY(i,j)] = in[IDY(i,j)]/factor;
			}
		}
		// Boundaries
		#pragma omp parallel for
		for (int i = 0; i < Nx; ++i) {
			out[IDY(i, 0)] = in[IDY(i,0)];
			out[IDY(i, Ny-1)] = in[IDY(i, Ny-1)];
		}
		#pragma omp parallel for
		for (int j = 0; j < Ny; ++j) {
			out[IDY(0, j)] = in[IDY(0, j)];
			out[IDY(Nx - 1, j)] = in[IDY(Nx - 1, j)];
		}
	}
}

/**
    @brief Function to impose boundary conditions to sub-domain or full domain array
    @param inout Array to impose BC
    @return void
*/
void SolverCG::ImposeBC(double* inout) {
	// Boundaries
	if (size > 1) {
		int k = 0;
		for (int i = x_start; i < 1+x_end; ++i) {
			for (int j = y_start; j < 1+y_end; ++j) {
				if ((j == 0) | (j == Ny-1) | (i == 0) | (i == Nx-1)) {
					inout[k] = 0.0;
				}
				k++;
			}
		}
	} else {
		#pragma omp parallel for
		for (int i = 0; i < Nx; ++i) {
			inout[IDY(i, 0)] = 0.0;
			inout[IDY(i, Ny-1)] = 0.0;
		}
		#pragma omp parallel for
		for (int j = 0; j < Ny; ++j) {
			inout[IDY(0, j)] = 0.0;
			inout[IDY(Nx - 1, j)] = 0.0;
		}
	}
}

/**
    @brief Function that converts sub-domain array to full domain array
    @param x Sub-domain array
    @param x_full Full domain array
    @return void
*/
void SolverCG::PartitionToDomain(double* x, double* x_full) {
	int k = 0;
	for (int i = x_start; i < 1+x_end; ++i) {
		for (int j = y_start; j < 1+y_end; ++j) {
			x_full[IDY(i,j)] = x[k];
			k++;
		}
	}
}

/**
    @brief Function to augment sub-domain array with an outer layer required for ApplyOperator calculations
    @param p Sub-domain array
    @param p_layered Layered sub-domain array with values from neigbouring sub-domains
    @return void
*/
void SolverCG::AddLayer(double* p, double* p_layered) {

    // Define arrays to send and recieve partition layers 
    double left_send[height];
    double right_send[height];
    double top_send[width];
    double bottom_send[width];
    double left_recv[height];
    double right_recv[height];
    double top_recv[width];
    double bottom_recv[width];

    /// Sending layers
    /// tags: left(0) right(1) top(2) bottom(3)
	
	// Left
    if (pos_x != 0) {
        for (int j = 0; j < height; ++j) {
            left_send[j] = p[j];
        }
	// Send left layer to rank to the left (rank-1) 
        MPI_Send(left_send, height, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD);
    }
    // Right
    if (pos_x != part-1) {
        for (int j = 0; j < height; ++j) {
            right_send[j] = p[(width-1)*height + j];
        }
	// Send right layer to rank to the right (rank+1)
        MPI_Send(right_send, height, MPI_DOUBLE, rank+1, 1, MPI_COMM_WORLD);
    }
	// Top 
    if (pos_y != part-1) {
        for (int i = 0; i < width; ++i) {
            top_send[i] = p[i*height + (height-1)];
        }
	// Send top layer to rank to the top (rank+p)
        MPI_Send(top_send, width, MPI_DOUBLE, rank+part, 2, MPI_COMM_WORLD);
    }
	// Bottom 
    if (pos_y != 0) {
        for (int i = 0; i < width; ++i) {
            bottom_send[i] = p[i*height];
        }
	// Send botttom layer to rank to the bottom (rank-p)
        MPI_Send(bottom_send, width, MPI_DOUBLE, rank-part, 3, MPI_COMM_WORLD);
    }
	
	/// Recieving layers
	/// tags: left(1) right(0) top(3) bottom(2)
	
	// Left
    if (pos_x != 0) {
	// Recieve new left layer (right layer wtr to other rank) from rank to the left (rank-1)
        MPI_Recv(left_recv, height, MPI_DOUBLE, rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < height; ++j) {
            p_layered[j+1] = left_recv[j];
        }
    }
    // Right
    if (pos_x != part-1) {
	// Recieve new rght layer (left layer wtr to other rank) from rank to the right (rank+1)
        MPI_Recv(right_recv, height, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < height; ++j) {
            p_layered[(width+1)*(height+2)+(j+1)] = right_recv[j];
        }
    }
    // Top
    if (pos_y != part-1) {
	// Recieve new top layer (bottom layer wtr to other rank) from rank to the top (rank+p)
        MPI_Recv(top_recv, width, MPI_DOUBLE, rank+part, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < width; ++i) {
            p_layered[(i+1)*(height+2)+(height+1)] = top_recv[i];
        }
    }
    // Bottom
    if (pos_y != 0) {
	// Recieve new bottom layer (top layer wtr to other rank) from rank to the bottom (rank-p)
        MPI_Recv(bottom_recv, width, MPI_DOUBLE, rank-part, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < width; ++i) {
            p_layered[(i+1)*(height+2)] = bottom_recv[i];
        }
    }
	
    // Once sub-domain layer created from neighbouring data, add initial sub-domain data to new array
    for (int i = 0; i < width; ++i) {
	for (int j = 0; j < height; ++j) {
	    p_layered[(i+1)*(height+2)+(j+1)] = p[i*height + j];
	}
    }
}

/**
    @brief Function to do dot product of two arrays (used to parallelise cblas_ddot with OpenMP)
    @param in1, in2 Arrays for dot product
    @return void
*/
double SolverCG::DotProd(double* in1, double* in2) {
	double result = 0.0;
	int i,j;
	#pragma omp parallel for reduction(+:result) 
	for (i = 0; i < Nx; ++i) {
		for (j = 0; j < Ny; ++j) {
			result += in1[IDY(i,j)]*in2[IDY(i,j)];
		}
	}
	return result;
}

/**
    @brief Function to copy array to another (used to parallelise cblas_dcopy with OpenMP)
    @param in Entering array to be copied
    @param out New copied array
    @return void
*/
void SolverCG::Copy(double* in, double* out) {
	int i,j;
	#pragma omp parallel for
	for (i = 0; i < Nx; ++i) {
		for (j = 0; j < Ny; ++j) {
			out[IDY(i,j)] = in[IDY(i,j)];
		}
	}
}

/**
    @brief Function that performs cblas_daxpy with for loops (used to parallelise cblas_daxpy with OpenMP)
    @param in Equivalent to x in daxpy
    @param out Equivalent to y in daxpy
    @param c Equivalent to alpha in daxpy 
    @return void
*/
void SolverCG::Daxpy(double* in, double* out, double c) {
	int i,j;
	#pragma omp parallel for 
	for (i = 0; i < Nx; ++i) {
		for (j = 0; j < Ny; ++j) {
			out[IDY(i,j)] = c*in[IDY(i,j)] + out[IDY(i,j)];
		}
	}
}
