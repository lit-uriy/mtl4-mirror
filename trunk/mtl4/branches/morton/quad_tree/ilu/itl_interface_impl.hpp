/*****************************************************************************
  file: itl_interface_impl.hpp
  ----------------------------
  Defines functions to be used with the ITL's GMRes function:
  	- mult: w = Ax + b
  	- solve  --> forward and backward subtitutions.

  For now, Make sure to pre-instantiate a global Matrix object called "mat",
  for bounds checking... This object holds all the data descriptive of the
  original matrix, that was destructively LU decomposed and whose quadtree
  representation's head node was passed into the solver (w/ type name Both)

  Created on: 08/13/06

  Larisse D. Voufo
  Peter Gottschling
*****************************************************************************/

#ifndef ITL_INTERFACE_IMP_HPP
#define ITL_INTERFACE_IMP_HPP

template <typename Vx, typename Vb, typename Vw>
void mult(	const Both& LUmat, indexType bIndex, int bc, int level,
			const Vx& x, int xI,
			const Vb& b, Vw& w, int bI);
template <typename Vb, typename Vy>
void for_solve(	const Both& LUmat, indexType bIndex, int bc, int level,
				const Vb& b, Vy& y, int bI);
template <typename Vy, typename Vx>
void back_solve(const Both& LUmat, indexType bIndex, int bc, int level,
				const Vy& y, Vx& x, int yI);

/*
Forward substitute
LUx = b <--> Ly = b
return y
*/
__attribute__((always inline))
template <typename Vb, typename Vy>
inline void for_solve(const Both& M, const Vb& b, Vy& y)
{
	for_solve<Vb,Vy>(M, MTN_START, BND_PART_ALL, LEVEL_START, b, y, 0);
}

/*
Backward substitute
LUx = b <--> Ly = b & Ux = y
return x
*/
__attribute__((always inline))
template <typename Vy, typename Vx>
inline void back_solve(const Both& M, const Vy& y, Vx& x)
{
	back_solve<Vy,Vx>(M, MTN_START, BND_PART_ALL, LEVEL_START, b, y, 0);
}


/*
Matrix-vector multiply
produces: w = Ax + b
*/
template <typename Vx, typename Vb, typename Vw>
void mult(	const Both& LUmat, indexType bIndex, int bc, int level,
			const Vx& x, int xI,
			const Vb& b, Vw& w, int bI)
{
	mat.bnd.endCheck(&bc, bIndex, level);
	if( bc == BND_OUT ) {
	}
	if(LUmat == NIL) {
	}
	else if(level < mat.maxLevel) { //is quad node
		bIndex *= 4;
		xI *= 2;
		bI *= 2;
		level++;
		mult( NW(LUmat), bIndex,   bc, level, x, xI,   b, w, bI  );
		mult( SW(LUmat), bIndex+1, bc, level, x, xI,   b, w, bI+1);
		mult( NE(LUmat), bIndex+2, bc, level, x, xI+1, b, w, bI  );
		mult( SE(LUmat), bIndex+3, bc, level, x, xI+1, b, w, bI+1);
	}
	else  {  // is base block
		int r, c, i, j, rI, iBO;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		bI *= baseOrder;
		xI *= baseOrder;
		for(i=0; i<r; i++) {
			rI = bI + i;
			iBO = i*baseOrder;
			for(j=0; j<c; j++){
				w[rI] = ((baseBlock)LUmat)[iBO+j] * x[xI+j] + b[rI];
			}
		}
	}
}


/*
Forward substitute
LUx = b <--> Ly = b
return y
*/
template <typename Vb, typename Vy>
void for_solve(	const Both& LUmat, indexType bIndex, int bc, int level,
				const Vb& b, Vy& y, int bI)
{
	mat.bnd.endCheck(&bc, bIndex, level);
	if( bc == BND_OUT ){
	}
	else if(LUmat == NIL){
		//diagonal matrix ==> y remains unchanged
	}
	else if(level < mat.maxLevel) {	//is quad node
		bI *= 2;
		level++;
		bIndex *= 4;
		for_solve<Vb,Vy>( 	NW(LUmat), bIndex, bc, level, b, y, bI );
		mult<Vy, Vb, Vy>( 	SW(LUmat), bIndex+1, bc, level,
							itl::scaled(y, -1.0), bI,
							b, y, bI+1);
		for_solve<Vy,Vy>( 	SE(LUmat), bIndex+3, bc, level, y, y, bI+1 );
	}
	else  {	//is base block
		int r, c, i, j, rI, iBO;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		bI *= baseOrder;
		for(i=0; i<r; i++) {
			rI = bI + i;
			iBO = i*baseOrder;
			y[rI]= b[rI];
			for(j=0; j<i; j++) {
				y[rI] -= ((baseBlock)LUmat)[iBO+j] * y[bI+j];
			}
		}
	}
}


/*
Backward substitute
LUx = b <--> Ly = b & Ux = y
return x
*/
template <typename Vy, typename Vx>
void back_solve(const Both& LUmat, indexType bIndex, int bc, int level,
				const Vy& y, Vx& x, int yI)
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
		yI *= 2;
		level++;
		bIndex *= 4;
		back_solve<Vy,Vx>( SE(LUmat), bIndex+3, bc, level, y, x, yI+1 );
		mult<Vx,Vy,Vx>( 	NE(LUmat), bIndex+2, bc, level,
							itl::scaled(x,-1.0), yI+1,
							y, x, yI );
		back_solve<Vx,Vx>( NW(LUmat), bIndex, bc, level, x, x, yI );
	}
	else  {	//is base block
		int r, c, i, j, rI, iBO;
		dataType D;
		mat.bnd.getEndBaseLimits(bc, &r, &c);
		rhsIndex *= baseOrder;
		for(i=r-1; i>=0; i--) {
			rI = yI + i;
			iBO = i*baseOrder;
			x[rI] = y[rI];
			for(j=r-1; j>i; j--) {
				x[rI] -= ((baseBlock)LUmat)[iBO+j] * x[yI+j] ;
			}
			D = ((baseBlock)LUmat)[iBO+i];
#if DENSE_DIAG
			if(D){
			   x[rI] /=  D;	//i==j
			}
			else {
		    //we should not be getting here
		    printf("\t backwardSubstitute-base: "
			         "Attempt to divide by zero. Exiting... bIndex= %d\n", bIndex);
		    exit(-17);
			}
#else
			if(D) {
				x[rI] /= ((baseBlock)LUmat)[iBO+j] ;	//i==j
			}
#endif
		}
	}
}



#endif // ITL_INTERFACE_IMP_HPP



//////////////////////////////////////////////////////////////////////////////
