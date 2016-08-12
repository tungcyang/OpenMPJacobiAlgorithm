// JacobiMethod.h -- header file for all the functions required for demonstrating Jacobi method as described in
// https://en.wikipedia.org/wiki/Jacobi_method.

#pragma once

#define MAX_NUM_ITERATIONS_4_CONV		1600		// The maximum number of iterations Jacobi method is allowed for attempting a convergence
													// of the solution.
#define MIN_JACOBI_DIMENSION			100			// Minimum and maximum n, where Ax = b with A being nxn.
#define MAX_JACOBI_DIMENSION			160
#define ATOL							1.0e-8		// A convergence of the solution has been reached when the error of the solution
													// is below this threshold between successive x.

void	CreateRandomDiagonallyDominantMatrix(double A[][MAX_JACOBI_DIMENSION], const int n);
void	MatrixMultiplication(double b[], const double A[][MAX_JACOBI_DIMENSION], const double x[], const int n);
int		SolveJacobiMethod(double xkp1[], const double A[][MAX_JACOBI_DIMENSION], const double b[], const int n);
