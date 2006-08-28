/*****************************************************************************
  file: sysSolve.cpp
  ------------------
  Solving system of linear equations: Ax=b
  fixed-point iteration using [? name of method ?]
  Reduce Error Correction and Increase convergence through damping.
  --> Need to choose the correct error convergence factor.
  		usually, less than 1 or 0.5 is OK.

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/

#include "sysSolve.h"


/*
void sysSolver(Both orig_mat, Both LUmat, dataType* res, int* iter)
	Performs the fix-point iteration untill acceptable result found
	params
		orig_mat: saved copy of original matrix
		LUmat:	matrix resulting from incomplete LU decomposition
				and holding the L and U preconditionners.
		res:	array of accumulated results.
		iter: 	number of iterations when solution is found
*/
void lu_sysSolver(Both orig_mat, Both LUmat, dataType* res, int* iter)
{
	int i, go=0;
	double b[mat.getRows()];
	*iter=1;
//printf("sysSolver:  %d\n", *iter);

	//backup rhs into b
	for( i=0; i<mat.getRows(); i++ )
		b[i] = mat.rhs[i];

	//forward substitution --> Ly = b ==> y, in rhs
	forwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//backward substitution --> Ux = y ==> x, in rhs
	backwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//initialize result
	for( i=0; i<mat.getRows(); i++ ) {
		res[i] = mat.rhs[i];
	}

	//rhs = b - Ax = residue
	//if(rhs != 0) then result += x', where Ax' = residue;
	if(errorEstimate(orig_mat, res, (dataType*)b)) {
		//do{
		// forward substitution --> Ly' = residue ==> y'
		// backward substitution --> Ux' = y' ==> x'
		// x += x'
		// b -= Ax
		//}while(b != 0)
		do {
			(*iter)++;				//increment number of iterations
//printf("sysSolver:  %d\n", *iter);
			forwardSubstitute(LUmat, MTN_START, BND_PART_ALL,LEVEL_START, 0);
			backwardSubstitute(LUmat, MTN_START, BND_PART_ALL,LEVEL_START, 0);
			//accumulate result
			for(i=0; i<mat.getRows(); i++) {
				res[i] += OMEGA * mat.rhs[i];
			}
		}while(errorEstimate(orig_mat, res, (dataType*)b));
	}
}

void ul_sysSolver(Both orig_mat, Both LUmat, dataType* res, int* iter)
{
	int i, go=0;
	double b[mat.getRows()];
	*iter=1;
//printf("sysSolver:  %d\n", *iter);

	//backup rhs into b
	for( i=0; i<mat.getRows(); i++ )
		b[i] = mat.rhs[i];

	//backward substitution --> Ux = y ==> x, in rhs
	backwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//forward substitution --> Ly = b ==> y, in rhs
	forwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//initialize result
	for( i=0; i<mat.getRows(); i++ ) {
		res[i] = mat.rhs[i];
	}

	//rhs = b - Ax = residue
	//if(rhs != 0) then result += x', where Ax' = residue;
	if(errorEstimate(orig_mat, res, (dataType*)b)) {
		//do{
		// forward substitution --> Ly' = residue ==> y'
		// backward substitution --> Ux' = y' ==> x'
		// x += x'
		// b -= Ax
		//}while(b != 0)
		do {
			(*iter)++;				//increment number of iterations
//printf("sysSolver:  %d\n", *iter);
			backwardSubstitute(LUmat, MTN_START, BND_PART_ALL,LEVEL_START, 0);
			forwardSubstitute(LUmat, MTN_START, BND_PART_ALL,LEVEL_START, 0);
			//accumulate result
			for(i=0; i<mat.getRows(); i++) {
				res[i] += OMEGA * mat.rhs[i];
			}
		}while(errorEstimate(orig_mat, res, (dataType*)b));
	}
}

/*
void forwardSubstitute(	Both LUmat,
						indexType bIndex, int bc, int level, int rhsIndex)
	forward substitute rhs into LUmat
*/
void forwardSubstitute(	Both LUmat,
						indexType bIndex, int bc, int level, int rhsIndex)
{
	mat.bnd.endCheck(&bc, bIndex, level);
	if( bc == BND_OUT ){
	}
	else if(LUmat == NIL){
		//diagonal matrix ==> y remains unchanged
	}
	else if(level < mat.maxLevel) {	//is quad node
		rhsIndex *= 2;
		level++;
		bIndex *= 4;
		forwardSubstitute( 	NW(LUmat), bIndex, bc, level, rhsIndex );
		mat_vect_mult( 	SW(LUmat), bIndex+1, bc, level, rhsIndex, rhsIndex+1);
		forwardSubstitute( 	SE(LUmat), bIndex+3, bc, level,   rhsIndex+1 );
	}
	else  {	//is base block
		int r, c, i, j, rI, iBO;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		rhsIndex *= baseOrder;
		for(i=0; i<r; i++) {
			rI = rhsIndex + i;
			iBO = i*baseOrder;
			for(j=0; j<i; j++) {
				mat.rhs[rI]-= ((baseBlock)LUmat)[iBO+j] * mat.rhs[rhsIndex+j];
			}
		}
	}
}

/*
void backwardSubstitute(	Both LUmat,
							indexType bIndex, int bc, int level, int rhsIndex)
	backward substitute rhs into LUmat
*/
void backwardSubstitute(	Both LUmat,
							indexType bIndex, int bc, int level, int rhsIndex)
{
	mat.bnd.endCheck(&bc, bIndex, level);
	if( bc == BND_OUT ) {
	}
	else if(LUmat == NIL) {
#if DENSE_DIAG
		//we should not be getting here, since det(NIL) = undefined.
		//   ==> NIL matrix has no inverse.
		printf("\t backwardSubstitute: "
			"Attempt to backward substitute into a NIL matrix. Exiting...\n");
		exit(-17);
#else
		//diagonal matrix ==> y remains unchanged
#endif
	}
	else if(level < mat.maxLevel) {	//is quad node
		rhsIndex *= 2;
		level++;
		bIndex *= 4;
		backwardSubstitute( SE(LUmat), bIndex+3, bc, level, rhsIndex+1 );
		mat_vect_mult( 	NE(LUmat), bIndex+2, bc, level, rhsIndex+1, rhsIndex );
		backwardSubstitute( NW(LUmat), bIndex, bc, level,   rhsIndex );
	}
	else  {	//is base block
		int r, c, i, j, rI, iBO;
		dataType D;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		rhsIndex *= baseOrder;
		for(i=r-1; i>=0; i--) {
			rI = rhsIndex + i;
			iBO = i*baseOrder;
			for(j=r-1; j>i; j--) {
				mat.rhs[rI] -= ((baseBlock)LUmat)[iBO+j] * mat.rhs[rhsIndex+j] ;
			}
			D = ((baseBlock)LUmat)[iBO+i];
#if DENSE_DIAG
			if(D){
			   mat.rhs[rI] /=  D;	//i==j
			}
			else {
		    //we should not be getting here
		    printf("\t backwardSubstitute-base: "
			         "Attempt to divide by zero. Exiting... bIndex= %d\n", bIndex);
		    exit(-17);
			}
#else
			if(D) {
				mat.rhs[rI] /= ((baseBlock)LUmat)[iBO+j] ;	//i==j
			}
#endif
		}
	}
}

/*
void mat_vect_mult( Both LUmat, indexType bIndex,
					int bc, int level, int rhsIn, int rhsOut)
	one part of rhs -= LUmat * another part of rhs;
	rhsIn --> starting index, in rhs, of the part that is multiplied in
	rhsOut --> starting index, in rhs, of the part that is updated.
	level --> level in tree
	bc --> bound cheching state
	LUmat --> LU decomposed matrix
*/
void mat_vect_mult( Both LUmat, indexType bIndex,
					int bc, int level, int rhsIn, int rhsOut)
{
	mat.bnd.endCheck(&bc, bIndex, level);
	if( bc == BND_OUT ) {
	}
	if(LUmat == NIL) {
	}
	else if(level<mat.maxLevel) /*(isQuadNode(LUmat))*/ {
		bIndex *= 4;
		rhsIn *= 2;
		rhsOut *= 2;
		level++;
		mat_vect_mult( NW(LUmat), bIndex, bc, level,   rhsIn, rhsOut);
		mat_vect_mult( SW(LUmat), bIndex+1, bc, level, rhsIn, rhsOut+1);
		mat_vect_mult( NE(LUmat), bIndex+2, bc, level, rhsIn+1, rhsOut);
		mat_vect_mult( SE(LUmat), bIndex+3, bc, level, rhsIn+1, rhsOut+1);
	}
	else /*if(isBaseBlk(LUmat))*/ {
		int r, c, i, j, rI, iBO;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		rhsOut *= baseOrder;
		rhsIn *= baseOrder;
		for(i=0; i<r; i++) {
			rI = rhsOut + i;
			iBO = i*baseOrder;
			for(j=0; j<c; j++){
				mat.rhs[rI] -= ((baseBlock)LUmat)[iBO+j] * mat.rhs[rhsIn+j];
			}
		}
	}
}

/*
void mat_vect_mult( Both LUmat, indexType bIndex, int bc, int level,
					dataType* res, int resI, int rhsI)
	rhs -= LUmat * res;
	rhsI --> starting index, in res
	rhsI --> starting index, in rhs
	res --> an array
	level --> level in tree
	bc --> bound cheching state
	bIndex --> current block's morton index
	LUmat --> LU decomposed matrix
*/
void mat_vect_mult( Both LUmat, indexType bIndex, int bc, int level,
					dataType* res, int resI, int rhsI)
{
	//bound checking
	mat.bnd.endCheck(&bc, bIndex, level);

	if( bc == BND_OUT ) {
	}
	if(LUmat == NIL) {
	}
	else if(level<mat.maxLevel) { //is quad node
		bIndex *= 4;
		resI *= 2;
		rhsI *= 2;
		level++;
		mat_vect_mult( NW(LUmat), bIndex,   bc, level, res, resI,   rhsI);
		mat_vect_mult( SW(LUmat), bIndex+1, bc, level, res, resI,   rhsI+1);
		mat_vect_mult( NE(LUmat), bIndex+2, bc, level, res, resI+1, rhsI);
		mat_vect_mult( SE(LUmat), bIndex+3, bc, level, res, resI+1, rhsI+1);
	}
	else {	//is base block
		int r, c, i, j, rI, iBO;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		rhsI *= baseOrder;
		resI *= baseOrder;
		for(i=0; i<r; i++) {
			rI = rhsI + i;
			iBO = i*baseOrder;
			for(j=0; j<c; j++)
				mat.rhs[rI] -= ((baseBlock)LUmat)[iBO+j] * res[resI+j] ;
		}
	}
}
/*
void readRhs()
	read in the rhs values from std in
	and store them in mat.rhs
*/
void readRhs()
{
	int i;
	for(i=0; i<mat.getRows(); i++)
		scanf("%lf",mat.rhs + i);
}

/*
int errorEstimate(	Both Mat, dataType* res, dataType* b )
	compute the current iteration's residue
		and return 1 if we are close enough to an acceptable result.
		return 0, if otherwise.
	The acceptable result is determined by the L-infinity norm of the
	rhs vector being <= a small error.
	L-inf norm + comparison by small eroor:
		--> For simplification of operation, compare the abs value of
			each element of rhs to the small error. Stop when an element
			with abs val > that small error is found; since the max
			number returned by the computation of the L-inf norm is
			>= to that number.
	b -= Mat * res
*/
int errorEstimate(	Both Mat, dataType* res, dataType* b )
{
	int i;
	//update matrix's rhs
	for(i=0; i<mat.getRows(); i++)
		mat.rhs[i] = b[i];
	//compute residue into rhs.
	mat_vect_mult(Mat, MTN_START, BND_PART_ALL, LEVEL_START, res, 0, 0);
	//are we close to the solution yet?
	//  --> is max abs(rhs[i]) <= SMALL_ERROR ?
	for(i=0; (i<mat.getRows()) && (abs((mat.rhs)[i]) <= SMALL_ERROR); i++);
	return (i<mat.getRows());
}


//---------------------------------------------------------------------


void lu_substitut_test(Both orig_mat, Both LUmat, dataType* res)
{
	int i, go=0;
	double b[mat.getRows()];

	//backup rhs into b
	for( i=0; i<mat.getRows(); i++ )
		b[i] = mat.rhs[i];

	//forward substitution --> Ly = b ==> y, in rhs
	forwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//backward substitution --> Ux = y ==> x, in rhs
	backwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//initialize result
	for( i=0; i<mat.getRows(); i++ ) {
		res[i] = mat.rhs[i];
	}

	//restore rhs
	for( i=0; i<mat.getRows(); i++ )
		mat.rhs[i] = b[i];
}

void ul_substitut_test(Both orig_mat, Both LUmat, dataType* res)
{
	int i, go=0;
	double b[mat.getRows()];

	//backup rhs into b
	for( i=0; i<mat.getRows(); i++ )
		b[i] = mat.rhs[i];

	//backward substitution --> Ux = y ==> x, in rhs
	backwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//forward substitution --> Ly = b ==> y, in rhs
	forwardSubstitute( LUmat, MTN_START, BND_PART_ALL, LEVEL_START, 0 );

	//initialize result
	for( i=0; i<mat.getRows(); i++ ) {
		res[i] = mat.rhs[i];
	}

	//restore rhs
	for( i=0; i<mat.getRows(); i++ )
		mat.rhs[i] = b[i];
}


//________________________________ END ______________________________________
/////////////////////////////////////////////////////////////////////////////

