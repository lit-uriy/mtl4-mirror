/*****************************************************************************
  file: iLU_up.cpp
  ----------------
  incomplete LU decomposition - produces preconditionners
  for iterative solutions for systems of linear equations

  LU Decomposing in the SE-NW direction

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/

#include "LU.h"

/*
parameters
  *bIndex:  current morton index
  sbc*:     starting bound checking status (northwest bound)
  ebc*:     ending bound checking status (southeast bound)
  level:    level in tree. starts at LEVEL_START (0)

starts each function with a bound check by calling mat.bnd.check()
  with the bound checking parameters.
stop if out of bounds.
*/

/*
void ULD(Both* Mat, indexType bIndex, int sbc, int ebc, int level)
  LU decompose (incompletely) the matrix represented by the node [Mat]
*/
void ULD(Both* Mat, indexType bIndex, int sbc, int ebc, int level)
{
  //bound checking
	mat.bnd.check(&sbc, &ebc, bIndex, level);

	//ULD
	if((sbc == BND_OUT) || (ebc == BND_OUT)){  //out of bounds
	}
	else if(*Mat == NIL){
#if DENSE_DIAG
		//non-singular subMatrix --> exit
		printf("\t ULD: Attempt to LU Decompose a NIL matrix. "
		          "Exiting...  bIndex= %d\n", bIndex);
		exit(-17);
#else
		//temp remains unchanged = NIL
#endif
	}
	else if(level < mat.maxLevel) {    // quad node
		level++;    //next level
		//quadrants' morton indices
		indexType bIndex_nw = bIndex*4;
		indexType bIndex_sw = bIndex_nw+1;
		indexType bIndex_ne = bIndex_nw+2;
		indexType bIndex_se = bIndex_nw+3;
		//LUD on se
		ULD(	&(SE(*Mat)), bIndex_se, sbc, ebc, level);
		//lower triSolve se into ne
		ul_lTS(	&(NE(*Mat)), bIndex_ne, sbc, ebc,
				    SE(*Mat),    bIndex_se, sbc, ebc, level);
		//upper triSolve se into sw
		ul_uTS(	&(SW(*Mat)), bIndex_sw, sbc, ebc,
				  SE(*Mat),    bIndex_se, sbc, ebc, level);
		//shur_dn complement sw and ne into nw
		ul_shur(  &(NW(*Mat)),
					    NE(*Mat), bIndex_ne, sbc, ebc,
              SW(*Mat), bIndex_sw, sbc, ebc, level);
		//LUD on se
		ULD(	&(NW(*Mat)), bIndex_nw, sbc, ebc, level);
	}
	else { //base block
		ULDbase( (baseBlock*)(Mat), bIndex, ebc);
	}
}

/*
void ul_uTS( 	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level )
  "upper" triangular solve: --> trisolve column-wise.
  The term "upper" reffers to the upper triangular part of the matrix [in]
    out = out * inverse(in)
*/
void ul_uTS( 	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level )
{
  //bound checking
	mat.bnd.check(&sbc_out, &ebc_out, bIndex_out, level);
	mat.bnd.check(&sbc_in, &ebc_in, bIndex_in, level);

  //ul_uTS
	if( (sbc_out==BND_OUT) || (ebc_out==BND_OUT) ||
		(sbc_in==BND_OUT) || (ebc_in==BND_OUT) ) {    //out of bounds
	}
	else if (in == NIL) {
#if DENSE_DIAG
		//we should not be getting here, since det(NIL) = undefined.
		//   ==> NIL matrix has no inverse.
		printf("\t ul_uTS: Attempt to triSolve from a NIL matrix. "
		          "Exiting...  bIndex= %d\n", bIndex_in);
		exit(-17);
#else
		//temp remains unchanged
#endif
	}
	else if(*out == NIL) {
		//*out remains unchanged
	}
	else if(level < mat.maxLevel) {  //is quad node
		//quadrants' morton indices
		indexType bIndex_out_nw = bIndex_out*4;
		indexType bIndex_out_sw = bIndex_out_nw+1;
		indexType bIndex_out_ne = bIndex_out_sw+1;
		indexType bIndex_out_se = bIndex_out_ne+1;
		indexType bIndex_in_nw = bIndex_in*4;
		indexType bIndex_in_ne = bIndex_in*4+2;
		indexType bIndex_in_se = bIndex_in_ne+1;
		level++;  //next level
		//upper TriSolve from SE of [in], into SE of [out]
		ul_uTS(	&(SE(*out)), bIndex_out_se, sbc_out, ebc_out,
				SE(in), bIndex_in_se, sbc_in, ebc_in, level );
		//upper TriSolve from SE of [in], into SW of [out]
		ul_uTS(	&(SW(*out)), bIndex_out_sw, sbc_out, ebc_out,
				SE(in), bIndex_in_se, sbc_in, ebc_in, level );
		//shur_up complement SE of [out], and NE of [in], into NE of [out]
		ul_shur(	&(NE(*out)),
				NE(in), bIndex_in_ne, sbc_in, ebc_in,
				SE(*out), bIndex_out_se, sbc_out, ebc_out, level );
		//shur_up complement SW of [out], and NE of [in], into NW of [out]
		ul_shur(	&(NW(*out)),
				NE(in), bIndex_in_ne, sbc_in, ebc_in,
				SW(*out), bIndex_out_sw, sbc_out, ebc_out, level );
		//upper TriSolve from NW of [in], into NE of [out]
		ul_uTS( 	&(NE(*out)), bIndex_out_ne, sbc_out, ebc_out,
				NW(in), bIndex_in_nw, sbc_in, ebc_in, level );				
		//upper TriSolve from NW of [in], into NW of [out]
		ul_uTS(	&(NW(*out)), bIndex_out_nw, sbc_out, ebc_out,
				NW(in), bIndex_in_nw, sbc_in, ebc_in, level );

		//set node to NIL, of all 4 children are NIL
		normalizeQuad((quadNode**)(out));
	}
	else { //is base block
		ul_uTSbase( (baseBlock*)(out), ebc_out, (baseBlock)(in), bIndex_in);
	}
}

/*
void ul_lTS( 	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level )
  "lower" triangular solve: --> trisolve row-wise
  The term "lower" reffers to the lower triangular part of the matrix [in],
  	filled with 1's at the diagonal.
    out = inverse( in ) * out
*/
void ul_lTS(	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level)
{
  //bound checking
	mat.bnd.check(&sbc_out, &ebc_out, bIndex_out, level);
	mat.bnd.check(&sbc_in, &ebc_in, bIndex_in, level);

  //ul_lTS
	if( (sbc_out==BND_OUT) || (ebc_out==BND_OUT) ||
		(sbc_in==BND_OUT) || (ebc_in==BND_OUT) ){
	}
	else if (in == NIL) {
#if DENSE_DIAG
		//we should not be getting here, since det(NIL) = undefined.
		//   ==> NIL matrix has no inverse.
		printf("\t ul_lTS: Attempt to triSolve from a NIL matrix. "
		          "Exiting...  bIndex= %d\n", bIndex_in);
		exit(-17);
#else
		//temp remains unchanged
#endif
	}
	else if(*out == NIL){
		//temp remains unchanged
	}
	else if(level <mat.maxLevel) { //is quad node
		//quadrants' morton indices
		indexType bIndex_out_nw = bIndex_out*4;
		indexType bIndex_out_sw = bIndex_out_nw+1;
		indexType bIndex_out_ne = bIndex_out_sw+1;
		indexType bIndex_out_se = bIndex_out_ne+1;
		indexType bIndex_in_nw = bIndex_in*4;
		indexType bIndex_in_sw = bIndex_in*4+1;
		indexType bIndex_in_se = bIndex_in_sw+2;
		level++;    //next level
		//lower TriSolve from SE of [in], into SE of [out]
		ul_lTS( &(SE(*out)), bIndex_out_se, sbc_out, ebc_out,
				SE(in), bIndex_in_se, sbc_in, ebc_in, level );
		//lower TriSolve from SE of [in], into NE of [out]
		ul_lTS( &(NE(*out)), bIndex_out_ne, sbc_out, ebc_out,
				SE(in), bIndex_in_se, sbc_in, ebc_in, level );
		//shur_up complement SE of [out], and SW of [in], into SW of [out]
		ul_shur( &(SW(*out)),
				SE(*out), bIndex_out_se, sbc_out, ebc_out,
				SW(in), bIndex_in_sw, sbc_in, ebc_in, level );
		//shur_up complement NE of [out], and SW of [in], into NW of [out]
		ul_shur( &(NW(*out)),
				NE(*out), bIndex_out_ne, sbc_out, ebc_out,
				SW(in), bIndex_in_sw, sbc_in, ebc_in, level );
		//lower TriSolve from NW of [in], into SW of [out]
		ul_lTS( &(SW(*out)), bIndex_out_sw, sbc_out, ebc_out,
				NW(in), bIndex_in_nw, sbc_in, ebc_in, level );
		//lower TriSolve from NW of [in], into NW of [out]
		ul_lTS( &(NW(*out)), bIndex_out_nw, sbc_out, ebc_out,
				NW(in), bIndex_in_nw, sbc_in, ebc_in, level );

		//set node to NIL, of all 4 children are NIL
		normalizeQuad((quadNode**)(out));
	}
	else { //is base block
		ul_lTSbase( (baseBlock*)(out), ebc_out, (baseBlock)in);
	}
}


/*
ul_up_SHUR and ul_dn_SHUR: --> dovetailing during shur complement
	out -= in1 * in2
*/
void ul_up_SHUR(	
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level )
{
	//bounds checking
	mat.bnd.check(&sbc_in1, &ebc_in1, bIndex_in1, level);
	mat.bnd.check(&sbc_in2, &ebc_in2, bIndex_in2, level);

	if( (sbc_in1==BND_OUT) || (ebc_in1==BND_OUT) ||
		(sbc_in2==BND_OUT) || (ebc_in2==BND_OUT) ){		//out of bounds
	}
#if LU_TEST
	else if( (in1 == NIL) || (in2 == NIL) ){
#else
	else if( (in1 == NIL) || (in2 == NIL) || (*out == NIL)  ){
#endif
		//newNode remains the same
	}
	else if(level <mat.maxLevel) { //is quad node
#if LU_TEST
		if(*out == NIL)
			*out = (Both)fromQuadNode();
#endif
		//quadrants morton indices
		indexType bIndex_in1_nw = bIndex_in1*4;
		indexType bIndex_in1_sw = bIndex_in1_nw+1;
		indexType bIndex_in1_ne = bIndex_in1_sw+1;
		indexType bIndex_in1_se = bIndex_in1_ne+1;
		indexType bIndex_in2_nw = bIndex_in2*4;
		indexType bIndex_in2_sw = bIndex_in2_nw+1;
		indexType bIndex_in2_ne = bIndex_in2_sw+1;
		indexType bIndex_in2_se = bIndex_in2_ne+1;
		level++;  //next level
		ul_up_SHUR(	&(NW(*out)),
					      NW(in1), bIndex_in1_nw, sbc_in1, ebc_in1,
					      NW(in2), bIndex_in2_nw, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(NW(*out)),
					      NE(in1), bIndex_in1_ne, sbc_in1, ebc_in1,
					      SW(in2), bIndex_in2_sw, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(SW(*out)),
					      SE(in1), bIndex_in1_se, sbc_in1, ebc_in1,
					      SW(in2), bIndex_in2_sw, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(SW(*out)),
					      SW(in1), bIndex_in1_sw, sbc_in1, ebc_in1,
					      NW(in2), bIndex_in2_nw, sbc_in2, ebc_in2, level );

		ul_dn_SHUR(	&(SE(*out)),
					      SW(in1), bIndex_in1_sw, sbc_in1, ebc_in1,
					      NE(in2), bIndex_in2_ne, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(SE(*out)),
					      SE(in1), bIndex_in1_se, sbc_in1, ebc_in1,
					      SE(in2), bIndex_in2_se, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(NE(*out)),
					      NE(in1), bIndex_in1_ne, sbc_in1, ebc_in1,
					      SE(in2), bIndex_in2_se, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(NE(*out)),
					      NW(in1), bIndex_in1_nw, sbc_in1, ebc_in1,
					      NE(in2), bIndex_in2_ne, sbc_in2, ebc_in2, level );

		normalizeQuad((quadNode**)(out));
	}
	else { //is base block
		shurBase(	(baseBlock*)(out),
						  (baseBlock)in1, ebc_in1,
						  (baseBlock)in2, ebc_in2 );
	}
}

void ul_dn_SHUR(	
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level )
{
  //bound checking
	mat.bnd.check(&sbc_in1, &ebc_in1, bIndex_in1, level);
	mat.bnd.check(&sbc_in2, &ebc_in2, bIndex_in2, level);

	if( (sbc_in1==BND_OUT) || (ebc_in1==BND_OUT) ||
		(sbc_in2==BND_OUT) || (ebc_in2==BND_OUT) ){		//out of bounds
	}
#if LU_TEST
	else if( (in1 == NIL) || (in2 == NIL) ){
#else
	else if( (in1 == NIL) || (in2 == NIL) || (*out == NIL)  ){
#endif
		//newNode remains the same
	}
	else if(level < mat.maxLevel) {  //is quad node
#if LU_TEST
		if(*out == NIL)
			*out = (Both)fromQuadNode();
#endif
		//quadrants morton indices
		indexType bIndex_in1_nw = bIndex_in1*4;
		indexType bIndex_in1_sw = bIndex_in1_nw+1;
		indexType bIndex_in1_ne = bIndex_in1_sw+1;
		indexType bIndex_in1_se = bIndex_in1_ne+1;
		indexType bIndex_in2_nw = bIndex_in2*4;
		indexType bIndex_in2_sw = bIndex_in2_nw+1;
		indexType bIndex_in2_ne = bIndex_in2_sw+1;
		indexType bIndex_in2_se = bIndex_in2_ne+1;
		level++;    //next level
		ul_dn_SHUR(	&(NE(*out)),
					      NW(in1), bIndex_in1_nw, sbc_in1, ebc_in1,
					      NE(in2), bIndex_in2_ne, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(NE(*out)),
					      NE(in1), bIndex_in1_ne, sbc_in1, ebc_in1,
					      SE(in2), bIndex_in2_se, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(SE(*out)),
					      SE(in1), bIndex_in1_se, sbc_in1, ebc_in1,
					      SE(in2), bIndex_in2_se, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(SE(*out)),
					      SW(in1), bIndex_in1_sw, sbc_in1, ebc_in1,
					      NE(in2), bIndex_in2_ne, sbc_in2, ebc_in2, level );

		ul_up_SHUR(	&(SW(*out)),
					      SW(in1), bIndex_in1_sw, sbc_in1, ebc_in1,
					      NW(in2), bIndex_in2_nw, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(SW(*out)),
					      SE(in1), bIndex_in1_se, sbc_in1, ebc_in1,
					      SW(in2), bIndex_in2_sw, sbc_in2, ebc_in2, level );
		ul_up_SHUR(	&(NW(*out)),
					      NE(in1), bIndex_in1_ne, sbc_in1, ebc_in1,
					      SW(in2), bIndex_in2_sw, sbc_in2, ebc_in2, level );
		ul_dn_SHUR(	&(NW(*out)),
					      NW(in1), bIndex_in1_nw, sbc_in1, ebc_in1,
					      NW(in2), bIndex_in2_nw, sbc_in2, ebc_in2, level );

		normalizeQuad((quadNode**)(out));
	}
	else {   // is base block
		shurBase(	(baseBlock*)(out),
						  (baseBlock)in1, ebc_in1,
						  (baseBlock)in2, ebc_in2 );
	}
}


//------------------------------- Base cases ----------------------

//ULD
void ULDbase(baseBlock* Mat, indexType bIndex, int bndChk)
{
	baseBlock temp;
	int i, j, k, kmin, iMax, jMax, ibo;
	dataType D;

	mat.bnd.getEndBaseLimits(bndChk, &iMax, &jMax);
	assert (iMax == jMax);	//Mat is a perfect square

	iMax--;
	for( i=iMax; i>=0; i-- ) {
		ibo = i*baseOrder;
		for( j=iMax-1; j>=0; j-- ) {
			kmin = max(i,j);
			for(k=iMax; k > kmin; k-- )
				(*Mat)[ibo+j] -= (*Mat)[ibo+k] * (*Mat)[k*baseOrder+j];
			if(i>j) {
				//D = (*Mat)[j*baseOrder+j];
				D = (*Mat)[ibo+i];
				if(D)
					(*Mat)[ibo+j] /= D;
				else {
#if DENSE_DIAG
					//non-singular subMatrix --> exit
					printf("\t ULDbase: NULL element at diagonal. "
		          "Exiting...  bIndex= %d\n", bIndex);
					exit(-17);
#else
					//temp remains unchanged = NIL
#endif
				}
			}
		}
	}
}

//triSolve column-wise
void ul_uTSbase(baseBlock* out, int bndChk_out, 
                  baseBlock in, indexType bIndex_in)
{
	int j, k, rows_out, cols_out, kbo, ibo;
	dataType D;
	mat.bnd.getEndBaseLimits(bndChk_out, &rows_out, &cols_out);
	rows_out--;
	cols_out--;
	for( k=rows_out; k>=0; k-- ) {
		kbo = k*baseOrder;
		D = in[kbo+k];
		if(!D){
#if DENSE_DIAG
			//non-singular subMatrix --> exit
			printf("\t ul_uTSbase: NULL element at diagonal. "
		          "Exiting...  bIndex= %d\n", bIndex_in);
			exit(-17);
#else
			//temp remains unchanged = NIL
			D = 1;
#endif
		}
		for( j=cols_out;  j>=0; j-- ) {
			(*out)[kbo+j] /= D;
		}
		for( ibo=kbo-baseOrder; ibo>=0; ibo-=baseOrder ) {
			for( j=cols_out;  j>=0; j-- ) {
				(*out)[ibo+j] -= (*out)[kbo+j] * in[ibo + k] ;
			}
		}
	}
	normalizeBBlock(out);
}

//triSolve row-wise
void ul_lTSbase(baseBlock* out, int bndChk_out, baseBlock in)
{
	int j, k, rows_out, cols_out, R, kbo, ibo;
	mat.bnd.getEndBaseLimits(bndChk_out, &rows_out, &cols_out);

	rows_out--;
	cols_out--;
	rows_out *= baseOrder;
	for( k=cols_out; k>=0; k-- ) {
		kbo = k*baseOrder;
		for( ibo=rows_out; ibo>=0; ibo-=baseOrder )
			for( j=k-1; j>=0; j-- )
				(*out)[ibo+j] -= in[kbo+j] * (*out)[ibo+k] ;
	}
	normalizeBBlock(out);
}





//________________________________ END ______________________________________
/////////////////////////////////////////////////////////////////////////////

