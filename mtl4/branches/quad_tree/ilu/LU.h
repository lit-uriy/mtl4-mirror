/*****************************************************************************
  file: iLU.h
  -----------
  incomplete LU decomposition - produces preconditionners
  for iterative solutions for systems of linear equations

  Revised on: 07/27/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef MATLUP_H
#define MATLUP_H

#include "matrix.h"

#define LU_TEST		1	   //set when needing to verify basic LUD algorithm,
                        	//without pivoting, for testing purposes


//---------------------------------------------------------------------------
// LUD
//---------------------------------------------------------------------------
void LUDbase(	baseBlock* Mat, indexType bIndex, int bndChk);
void lu_uTSbase(	baseBlock* out, int bndChk_out,
                  baseBlock in, indexType bIndex_in);
void lu_lTSbase(	baseBlock* out, int bndChk_out, baseBlock in);
void shurBase(	baseBlock* out,
					baseBlock in1, int bndChk_in1,
					baseBlock in2, int bndChk_in2);
//----------------------------------------
void LUD(	Both* Mat, indexType bIndex, int sbc, int ebc, int level);
void lu_uTS(	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level);
void lu_lTS(	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level);
void ul_up_SHUR(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level);
void lu_dn_SHUR(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level);

__attribute__ ((always_inline))
inline void lu_shur(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level)
{
	lu_dn_SHUR(	 out,
					     in1, bIndex_in1, sbc_in1, ebc_in1,
					     in2, bIndex_in2, sbc_in2, ebc_in2, level );
}

//---------------------------------------------------------------------------
// ULD
//---------------------------------------------------------------------------
void ULDbase(	baseBlock* Mat, indexType bIndex, int bndChk);
void ul_uTSbase(	baseBlock* out, int bndChk_out,
                  baseBlock in, indexType bIndex_in);
void ul_lTSbase(	baseBlock* out, int bndChk_out, baseBlock in);
//----------------------------------------
void ULD(	Both* Mat, indexType bIndex, int sbc, int ebc, int level);
void ul_uTS(	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level);
void ul_lTS(	Both* out, indexType bIndex_out, int sbc_out, int ebc_out,
 			Both in, indexType bIndex_in, int sbc_in, int ebc_in, int level);
void ul_up_SHUR(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level);
void ul_dn_SHUR(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level);

__attribute__ ((always_inline))
inline void ul_shur(
		Both* out,
		Both in1, indexType bIndex_in1, int sbc_in1, int ebc_in1,
		Both in2, indexType bIndex_in2, int sbc_in2, int ebc_in2, int level )
{
	ul_up_SHUR(	 out,
					     in1, bIndex_in1, sbc_in1, ebc_in1,
					     in2, bIndex_in2, sbc_in2, ebc_in2, level );
}


//---------------------------------------------------------------------------

#endif
/////////////////////////////////////////////////////////////////////////////

