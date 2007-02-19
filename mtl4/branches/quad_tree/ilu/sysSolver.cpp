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

#include "sysSolver.h"

/* MAIN
*/
int main(int argc, char** argv)
{
  env_init(argc, argv);
	readRhs();
	
	//solve system of equations
	solveSys();

  env_kill();
  
	return 0;
}


//________________________________ END ____________________________________________________
///////////////////////////////////////////////////////////////////////////////////////////

