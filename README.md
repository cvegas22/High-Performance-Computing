# High-Performance Computing Coursework
**Parallelisation and Optimisation of a Lid-Driven Cavity Solver**

---

## Overview
This project contains an optimised and parallelised C++ implementation for the lid-driven cavity flow problem, solving the incompressible Navier-Stokes equations in two dimensions using a finite difference approach. Parallelism is achieved through **MPI** for distributed-memory parallelisation and **OpenMP** for shared-memory threading. The numerical solver employs a preconditioned **Conjugate Gradient (CG)** method to solve the Poisson equation iteratively.

---

## Project Structure
```bash
.
├── LidDrivenCavity.cpp/h            # Core class for setting up and solving the problem
├── LidDrivenCavitySolver.cpp        # Main driver program
├── SolverCG.cpp/h                   # Conjugate Gradient solver class with MPI/OpenMP support
├── unit_test_LidDrivenCavity.cpp    # Unit test for LidDrivenCavity class
├── unit_test_SolverCG.cpp           # Unit test for SolverCG class
├── Makefile                         # Compilation and testing commands
├── Doxyfile                         # Configuration file for documentation generation (Doxygen)
├── repository.log                   # Git commit history
└── README.md                        # This README file
```

## Dependencies
- MPI: Distributed-memory parallel execution.
- OpenMP: Shared-memory threading.
- CBLAS: BLAS routines for linear algebra operations.
- Boost: Program options parsing and unit testing (Boost.Test).
- Doxygen: Automated documentation generation.

## Compilation
Use the provided Makefile to compile:
```bash
make                 # Compiles solver executable (named 'solver')
make doc             # Generates documentation using Doxygen
make unittests       # Compiles and runs unit tests
```

## Running the Solver
Execute the solver with command-line parameters, e.g.:
```bash
mpirun -np 4 ./solver --Lx 1 --Ly 1 --Nx 200 --Ny 200 --Re 1000 --dt 0.005 --T 50 --p 2
```

### Available Command-line Arguments:
- --Lx: Domain length in the x-direction (default: 1)
- --Ly: Domain length in the y-direction (default: 1)
- --Nx: Grid points in the x-direction (default: 9)
- --Ny: Grid points in the y-direction (default: 9)
- --p: Partitions per direction for MPI (must satisfy p*p MPI ranks)
- --dt: Time step size (default: 0.01)
- --T: Final simulation time (default: 0.1)
- --Re: Reynolds number (default: 10)
- --verbose: Enable verbose output
- --help: Show help message

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
