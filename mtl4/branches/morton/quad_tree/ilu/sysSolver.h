/*****************************************************************************
  file: sysSolver.cpp
  -----------------
  Solving system of linear equations: Ax=b
  	--> incomplete LU +
  		fixed point iteration w/ forward and backward substitutions

  contains Main function

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef SYSSOLVE_H
#define SYSSOLVE_H

#include "env_setup.h"
#include "LU.h"
#include "sysSolve.h"

/*
void solveSys()
	Solves the system of linear equations:
	- backs up a copy of the original matrix before performing the incomplete
		LU decomposition since this is done destructively.
	- Then, calls the iterative solver...
	- Finally, checks the result and print out the number of iterations needed
		to find the solution.
	Timing is performed only during inc LUD and iteration.
*/
static void solveSys()
{
	//prepare the array that will hold the final result
	dataType* result = (dataType*)(valloc(mat.getRows() * sizeof(dataType)));
	int iter;

	//save a copy of original matrix
	mat.copyMatrix(&(mat.backupNode), mat.mainNode, LEVEL_START);

	//print input
	if(printInput){
		printf("\nMatrix:\n");
		printMatrix(mat.mainNode, printINformat);
		printf("\n");

		printf("\nRhs:\n");
		printArray(mat.rhs, mat.getRows() );
		printf("\n");
  }

	timestamp_start("");
		//LU Decomposition with pivoting
mat.numb_fill_in = 0;
printf(" Number of baseblocks - start:  %d\n", numb_bblocks);
		LUD( &(mat.mainNode),
				MTN_START, BND_PART_ALL, BND_PART_ALL , LEVEL_START );
printf(" Number of fill ins generated:  %d\n", mat.numb_fill_in);	
printf(" Number of baseblocks - end:  %d\n", numb_bblocks);
	/*			
		printf("\nMatrix:\n");
		printMatrix(mat.mainNode, printINformat);
		printf("\n");
	*/	
		
	//timestamp_end("LU Decomposing - TIME TAKEN:");
		//Solve Ax = b
		lu_sysSolver(mat.backupNode, mat.mainNode, result, (int*)(&iter));
	timestamp_end("SOLVING SYS OF EQs. - TIME TAKEN:");

	int nans = 0;
	for(int i=0; i<mat.getRows(); i++){
		if(isnan(result[i]))
			nans++;
	}

	//print result and number of iterations
	printf(
		"\n\tResult returned after %d iterations - nans: %d\n\n", iter, nans);
	if(printOutput){
		printArray(result, mat.getRows() );
		printf("\n\n");
	}
	free(result);
}


#endif
//________________________________ END ____________________________________________________
///////////////////////////////////////////////////////////////////////////////////////////

