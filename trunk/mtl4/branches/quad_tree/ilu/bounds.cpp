/*****************************************************************************
  file: bounds.cpp
  ---------------- 
  Defines the two main sets functions described in bound.h 

  Revised on: 07/26/06
  
  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/

#include "bounds.h"

/*
parameters: 
  bndChk: Current bound checking state
           --> is the parent block currently fully within the bounds (BND_IN),
                or partially in rowwise(BND_PART_ROW), 
                or partially in columnwise(BND_PART_COL),
                or partially in either way(BND_PART_ALL),
                or fully outide of bounds (BND_OUT) ?
  mbIndex: morton Index of the current block.
  dRowBnd: dilated row index of the bound.
  dColBnd: dilated column index of the bound.
  level: level in tree.
return new bound checking state in bndChk
*/

//............................................................................
/*
BND_CHK_start:
  Is current block to the southeast of the bounds [dRowBnd, dColBnd] ?
  Determine right function to call (or not call) based on the 
    current bound checking state. 
*/
void MatBounds::BND_CHK_start(int* bndChk, indexType mbIndex, 
                              indexType dRowBnd, indexType dColBnd, int level)
{
	if(*bndChk == BND_IN){}
	else if(*bndChk == BND_PART_ALL)
		*bndChk = BND_CHK_start_All(mbIndex, dRowBnd, dColBnd, level);
	else if(*bndChk == BND_PART_ROW)
		*bndChk = BND_CHK_start_Row(mbIndex, dRowBnd, level);
	else if(*bndChk == BND_PART_COL)
		*bndChk = BND_CHK_start_Col(mbIndex, dColBnd, level);
	else;
}
int MatBounds::BND_CHK_start_All( indexType mbIndex, indexType dRowBnd, 
                                              indexType dColBnd, int level)
{
	indexType mbRow = mbIndex & evenBits;
	indexType mbCol = mbIndex & oddBits;
	if ( (mbRow >= dRowBnd) && (mbCol >= dColBnd) ) {
		return BND_IN;
	}
	else{
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		indexType rbIndex = mbIndex & evenBits;
		indexType cbIndex = mbIndex & oddBits;
		if( (rbIndex < dRowBnd) || (cbIndex < dColBnd) )
			return BND_OUT;
		else if(mbRow >= dRowBnd)
			return BND_PART_COL;
		else if(mbCol >= dColBnd)
			return BND_PART_ROW;
		else
			return BND_PART_ALL;
	}
}
int MatBounds::BND_CHK_start_Row( indexType mbIndex, 
                                  indexType dRowBnd, int level)
{
	if((mbIndex & evenBits) >= dRowBnd) {
		return BND_IN;
	}
	else{
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		return ( ((mbIndex & evenBits) < dRowBnd) ? BND_OUT : BND_PART_ROW );
	}
}
int MatBounds::BND_CHK_start_Col( indexType mbIndex, 
                                  indexType dColBnd, int level)
{
	if((mbIndex & oddBits) >= dColBnd) {
		return BND_IN;
	}
	else{
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		return ( ((mbIndex & oddBits) < dColBnd) ? BND_OUT : BND_PART_COL );
	}
}

//............................................................................
/*
BND_CHK_start:
  Is current block to the northwest of the bounds [dRowBnd, dColBnd] ?
  Determine right function to call (or not call) based on the 
    current bound checking state. 
*/
void MatBounds::BND_CHK_end(int* bndChk, indexType mbIndex, 
                            indexType dRowBnd, indexType dColBnd, int level)
{
	if(*bndChk == BND_IN){}
	else if(*bndChk == BND_PART_ALL)
		*bndChk = BND_CHK_end_All(mbIndex, dRowBnd, dColBnd, level);
	else if(*bndChk == BND_PART_ROW)
		*bndChk = BND_CHK_end_Row(mbIndex, dRowBnd, level);
	else if(*bndChk == BND_PART_COL)
		*bndChk = BND_CHK_end_Col(mbIndex, dColBnd, level);
	else;
}
int MatBounds::BND_CHK_end_All(indexType mbIndex, 
                              indexType dRowBnd, indexType dColBnd, int level)
{
	if ( ((mbIndex & evenBits) < dRowBnd) && ((mbIndex & oddBits) < dColBnd) ) {
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		indexType rbIndex = mbIndex & evenBits;
		indexType cbIndex = mbIndex & oddBits;
		if( (rbIndex < dRowBnd) && (cbIndex < dColBnd) )
			return BND_IN;
		else if(rbIndex < dRowBnd)
			return BND_PART_COL;
		else if(cbIndex < dColBnd)
			return BND_PART_ROW;
		else
			return BND_PART_ALL;
	}
	else{
		return BND_OUT;
	}
}
int MatBounds::BND_CHK_end_Row(indexType mbIndex, 
                               indexType dRowBnd, int level)
{
	if((mbIndex & evenBits) < dRowBnd) {
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		return ( ((mbIndex & evenBits) < dRowBnd) ? BND_IN : BND_PART_ROW );
	}
	else{
		return BND_OUT;
	}
}
int MatBounds::BND_CHK_end_Col(indexType mbIndex, 
                                indexType dColBnd, int level)
{
	if((mbIndex & oddBits) < dColBnd) {
		//block inside boundaries: fully or partially?
		mbIndex += (bSize[level]-1);
		return ( ((mbIndex & oddBits) < dColBnd) ? BND_IN : BND_PART_COL );
	}
	else{
		return BND_OUT;
	}
}

//________________________________ END _______________________________________
//////////////////////////////////////////////////////////////////////////////
