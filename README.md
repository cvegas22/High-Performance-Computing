# High-Performance Computing
**Parallelisation and Optimisation of Lid-Driven Cavity Solver**

---

## Project Overview
This repository provides a highly optimised and parallelised implementation of a solver for the 2D lid-driven cavity flow problem, using the vorticity-stream function formulation of the incompressible Navier-Stokes equations. Parallelisation was achieved using **MPI** for distributed-memory computing and **OpenMP** for shared-memory multi-threading. The solver uses a preconditioned **Conjugate Gradient (CG)** method for solving Poisson equations iteratively.

---

## Directory Structure
```bash
.
├── LidDrivenCavity.cpp/h            # Lid-driven cavity solver implementation
├── LidDrivenCavitySolver.cpp        # Main execution driver
├── SolverCG.cpp/h                   # Parallelised Conjugate Gradient solver
├── unit_test_LidDrivenCavity.cpp    # Unit tests for LidDrivenCavity class
├── unit_test_SolverCG.cpp           # Unit tests for SolverCG class
├── Makefile                         # Compilation commands
├── Doxyfile                         # Configuration for Doxygen documentation
├── repository.log                   # Detailed Git commit history
├── HPC_Report.pdf                   # Coursework report (MPI/OpenMP scaling and optimisations)
└── README.md                        # This file
```

## Dependencies
- MPI (OpenMPI or similar)
- OpenMP
- CBLAS (for optimized linear algebra computations)
- Boost libraries (program options parsing, Boost.Test for unit testing)
- Doxygen (documentation generation)

## Compilation
Use the provided Makefile to compile:

make                 # Compile solver executable named 'solver'
make doc             # Generate documentation via Doxygen
make unittests       # Compile and execute unit tests

## Execution
Run the solver with desired parameters, e.g.:

mpirun -np 4 ./solver --Lx 1 --Ly 1 --Nx 200 --Ny 200 --Re 1000 --dt 0.005 --T 50 --p 2

### Command-line Options:
- --Lx, --Ly: Domain dimensions (default: 1)
- --Nx, --Ny: Grid resolution (default: 9)
- --Re: Reynolds number (default: 10)
- --dt: Time step (default: 0.01)
- --T: Simulation end-time (default: 0.1)
- --p: Domain partitioning factor (must satisfy p*p MPI ranks)
- --verbose: Verbose output
- --help: Display help message

## MPI Parallelisation
- Domain decomposition into p² sub-domains, assigning one to each MPI process.
- Efficient communication by exchanging only boundary layers between neighboring ranks.
- MPI_Allreduce used for synchronizing intermediate and final results across processes.
- Good parallel scaling achieved with larger grid sizes (optimal for grids ≥ 100×100).

### MPI Performance Results (Runtime in seconds):
Grid Size | 1 Rank | 4 Ranks | 9 Ranks | 16 Ranks
25×25     | 3.1    | 5.3     | 7.9     | 8.2
50×50     | 31.7   | 20.8    | 24.2    | 24.4
100×100   | 259.2  | 108.0   | 91.5    | 72.2
200×200   | 2666   | 747.1   | 460.7   | 335.2
400×400   | 2180   | 588.2   | 287.9   | 197.0

## OpenMP Parallelisation
- Shared-memory threading implemented for computational loops in SolverCG.
- Custom parallel versions of cblas_ddot, cblas_daxpy, and cblas_dcopy implemented using OpenMP due to better performance over the standard CBLAS versions.

### OpenMP Performance Results (Runtime in seconds):
Threads | 100×100 | 200×200 | 400×400
1       | 259.2   | 2711    | 1874
4       | 203.1   | 838.8   | 629.7
8       | 177.4   | 705.1   | 363.9
16      | 293.8   | 765.0   | 272.6

Best performance noted for larger grid sizes and moderate thread counts (8 threads recommended).

## Optimisation and Profiling
Detailed profiling (using serial and single-threaded runs) identified critical computational hotspots:

- Major hotspots: BLAS operations (cblas_ddot, cblas_dnrm2), ApplyOperator, and Precondition functions.

Optimisation strategies included:
- Compiler optimisation flag improved from -O2 to -O3 (~10% runtime improvement).
- Removed redundant BLAS operations (21% additional runtime improvement).
- Reduced frequency of convergence checks (runtime reduced by additional ~15%).

### Optimisation Impact:
Total runtime reduced by ~40% (from 2713 s to 1638 s for 200×200 grid case).
