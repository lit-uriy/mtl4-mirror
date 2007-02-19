/****************************************************************************
  file: sysSolve.h
  ----------------
  Solving system of linear equations: Ax=b
  fixed-point iteration using [? name of method ?]
  Reduce Error Correction and Increase convergence through damping.
  --> Need to choose the correct error convergence factor.
  		usually, less than 1 or 0.5 is OK.

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
****************************************************************************/
#ifndef SYSSOLVE_H
#define SYSSOLVE_H

#include "matrix.h"

extern double OMEGA;		  //damping factor

void lu_sysSolver(Both orig_mat, Both LUmat, dataType* res, int* iter);
void ul_sysSolver(Both orig_mat, Both LUmat, dataType* res, int* iter);
void forwardSubstitute(	Both LUmat,
						indexType bIndex, int bc, int level, int rhsIndex);
void backwardSubstitute(Both LUmat,
						indexType bIndex, int bc, int level, int rhsIndex);
void mat_vect_mult( Both LUmat, indexType bIndex,
					int bc, int level, int rhsIn, int rhsOut);
void mat_vect_mult( Both LUmat, indexType bIndex,
					int bc, int level, dataType* res, int resI, int rhsI);
void readRhs();
int errorEstimate(	Both mat, dataType* res, dataType* b );

//---------------------------------------------------------------------
void lu_substitut_test(Both orig_mat, Both LUmat, dataType* res);
void ul_substitut_test(Both orig_mat, Both LUmat, dataType* res);

#endif
/////////////////////////////////////////////////////////////////////////////

