// JacobiMethod.cpp -- source file for all the functions required for demonstrating Jacobi method as described in
// https://en.wikipedia.org/wiki/Jacobi_method.

#include <stdlib.h>
#include <math.h>
#ifdef _OPENMP
#	include <omp.h>
#endif
#include "JacobiMethod.h"

#	include <stdio.h>

// SolveJacobiMethod() applies the Jacobi method (https://en.wikipedia.org/wiki/Jacobi_method) to solve the system of
// equations Ax = b.  It returns the number of iterations taken to reach the convergence of xk (capped by
// MAX_NUM_ITERATIONS_4_CONV.
int		SolveJacobiMethod(double xkp1[], const double A[][MAX_JACOBI_DIMENSION], const double b[], const int n)
{
	int		i, j, iIteration;
	double	xk[MAX_JACOBI_DIMENSION];
	double	fxErr;

	// Initializaing xk[] as the seed.
	for (i = 0; i < n; i++)
		xk[i] = 0.0;

	for (iIteration = 0; iIteration < MAX_NUM_ITERATIONS_4_CONV; iIteration++)
	{
		// xkp1[i] = (1.0/A[i][i])*(b[i] - Sum(j!=i) A[i][j]*xk[j]).  This is the loop where we want to experiment on OpenMP
		// effect.
		fxErr = 0.0;

#ifdef _OPENMP
		// In general we do not need to use private() nor shared().  All variables are considered shared except for the index
		// variables (i) and the variables declared in the parallel regions
		// (http://sc.tamu.edu/shortcourses/SC-openmp/OpenMPSlides_tamu_sc.pdf).
#pragma omp parallel for reduction(+ : fxErr)
#endif
		for (i = 0; i < n; i++)
		{
			double	fSum, fDiff;

			fSum = b[i];
			// We will first evaluate b[] - Sum A[][]*xk[] for all entries on the ith row and then add back the product for j == i;
			for (j = 0; j < n; j++)
				fSum -= A[i][j] * xk[j];
			fSum += A[i][i] * xk[i];
			xkp1[i] = fSum / A[i][i];

			fDiff = xkp1[i] - xk[i];
			fxErr += fDiff*fDiff;
		}
#ifdef _OPENMP
#pragma omp barrier
#endif

		// Have we converged enough to break out of the iteration before MAX_NUM_ITERATIONS_4_CONV?  Evaluate the
		// vector norm of (xkp1[] - xk[]).  We break out of the iteration if the difference norm is small enough.
		if (fxErr < ATOL)
			break;
		else
		{
			// We did not break the iteration.  Copying xkp1[] to xk[] and starting up again.
			for (i = 0; i < n; i++)
				xk[i] = xkp1[i];
		}
	}

	return iIteration;
}


// MatrixMultiplication() carries out the matrix multiplication, Ax = b.
void	MatrixMultiplication(double b[], const double A[][MAX_JACOBI_DIMENSION], const double x[], const int n)
{
	int		i, j;

	for (i = 0; i < n; i++)
	{
		b[i] = 0.0;
		for (j = 0; j < n; j++)
			b[i] += A[i][j] * x[j];
	}
}


// CreateRandomDiagonallyDominantMatrix() will generate an nxn random matrix A[][] which is diagonally dominant.
void	CreateRandomDiagonallyDominantMatrix(double A[][MAX_JACOBI_DIMENSION], const int n)
{
	int		i, j;
	double	fOffDiagonalSum;

	// Populating A[][]
	for (i = 0; i < n; i++)
		for (j = 0; j < n; j++)
			A[i][j] = (float) rand() / RAND_MAX;

	// Adjusting diagonal entries of A[][] so that it is diagonally dominant.
	// https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
	for (i = 0; i < n; i++)
	{
		fOffDiagonalSum = 0.0;
		for (j = 0; j < n; j++)
			if (j != i)
				fOffDiagonalSum += fabs(A[i][j]);

		// Modifying the diagonal entries.  We add or subtract the sum of absolute values of the off-diagonal entries from
		// the diagonal with the goal to increase the absolute value of the diagonal entries.
		if (A[i][i] > 0.0)
			A[i][i] += fOffDiagonalSum;
		else
			A[i][i] -= fOffDiagonalSum;

		// Another pass in case A[i][i] is too small which might make it somewhat singular.  We add 2.0 to A[i][i] if it
		// is small.
		if (fabs(A[i][i]) < 0.005)
		{
			if (A[i][i] >= 0.0)
				A[i][i] += 2.0;
			else
				A[i][i] -= 2.0;
		}
	}
}
