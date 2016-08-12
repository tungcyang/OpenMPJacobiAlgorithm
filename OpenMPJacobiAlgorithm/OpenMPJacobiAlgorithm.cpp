// OpenMPJacobiAlgorithm.cpp : Defines the entry point for the console application.
//

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#ifdef _OPENMP
#	include <omp.h>
#endif
#include "JacobiMethod.h"

#define NUM_ITERATION_JACOBI_METHOD			8192	// Number of iterations to carry out Jacobi method to solve the system
													// of equations.

													// Function elapsedTime() evaluates the elapsed time between two DWORD timing measurements as returned
													// from GetTickCount().  It takes into account the possible wraparound happening between the two
													// GetTickCount() calls.

// Function elapsedTime() evaluates the elapsed time between two DWORD timing measurements as returned
// from GetTickCount().  It takes into account the possible wraparound happening between the two
// GetTickCount() calls.
DWORD	elapsedTime(DWORD dStartTime, DWORD dStopTime)
{
	if (dStopTime > dStartTime)
		return (dStopTime - dStartTime);
	else
	{
		// A wraparound occurred between the two GetTickCount() calls.
		return (dStopTime + (MAXDWORD - dStartTime) + 1);
	}
}

int main()
{
	int			i;
	double		fAccumulatedSolErrors = 0.0;
	int			iAccumulatedIterations = 0, iMaxIteration = 0;
	DWORD		dTotalMilliSecElapsed = 0;
	
	// Initialization and resetting the random seed.
	srand(0);
#ifdef _OPENMP
	// OpenMP references:
	// "OpenMP in Visual C++" in https://msdn.microsoft.com/en-us/library/tt15eb9t.aspx
	// "OpenMP Directives" in https://msdn.microsoft.com/en-us/library/0ca2w8dk.aspx
	fprintf(stdout, "Number of processor cores for OpenMP Jacobi method: %d\n", omp_get_num_procs());
	fprintf(stdout, "Number of threads used for OpenMP Jacobi method: %d\n", omp_get_max_threads());
#endif

	for (i = 0; i < NUM_ITERATION_JACOBI_METHOD; i++)
	{
		int		iNumIterationsTaken;
		DWORD	dBeforeTimeStamp, dAfterTimeStamp;

		{
			int		i, n;
			double	A[MAX_JACOBI_DIMENSION][MAX_JACOBI_DIMENSION];
			double	b[MAX_JACOBI_DIMENSION];
			double	xk[MAX_JACOBI_DIMENSION];
			double	xsol[MAX_JACOBI_DIMENSION];
			double	fxErr = 0.0, fDiff;

			// Creating Ax = b.  We first generate diagonally dominant A and x, use them to compute b, and then use A and b to solve x.
			// Note that when deciding whether to stop iterations in Jacobi method, we do not know xsol.
			n = MIN_JACOBI_DIMENSION + (int)((float)rand() / RAND_MAX * (MAX_JACOBI_DIMENSION - MIN_JACOBI_DIMENSION + 1));
			CreateRandomDiagonallyDominantMatrix(A, n);
			for (i = 0; i < n; i++)
				xsol[i] = (float) rand() / RAND_MAX;
			MatrixMultiplication(b, A, xsol, n);

			// Now we have A and b, and we will solve for x so that Ax = b with Jacobi method.
			dBeforeTimeStamp = GetTickCount();
			iNumIterationsTaken = SolveJacobiMethod(xk, A, b, n);
			dAfterTimeStamp = GetTickCount();
			dTotalMilliSecElapsed += elapsedTime(dBeforeTimeStamp, dAfterTimeStamp);

			// Evaluating the error in the solution.
			for (i = 0; i < n; i++)
			{
				fDiff = xk[i] - xsol[i];
				fxErr = fDiff*fDiff;
			}

			fAccumulatedSolErrors += fxErr;
		}

		iAccumulatedIterations += iNumIterationsTaken;
		if (iMaxIteration < iNumIterationsTaken)
			iMaxIteration = iNumIterationsTaken;
	}

	fprintf(stdout, "After %d runs of Jacobi Methods, the accumulated errors in the solution is\n\t%e\n",
			NUM_ITERATION_JACOBI_METHOD, fAccumulatedSolErrors);
	fprintf(stdout, "On average Jacobi method required %f iterations to reach the solution.\n",
			(float) iAccumulatedIterations / NUM_ITERATION_JACOBI_METHOD);
	fprintf(stdout, "At most Jacobi method required %d iterations to solve the system.\n",
			iMaxIteration);
	fprintf(stdout, "All Jacobi method runs took %10.7f seconds to run.\n",
			(float) dTotalMilliSecElapsed/1000.0);

    return 0;
}

