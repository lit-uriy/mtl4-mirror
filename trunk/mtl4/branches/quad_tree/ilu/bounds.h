/*****************************************************************************
  file: bounds.h
  -------------- 
  Defines MatBounds class: bounds checking environment for a given matrix.

  Revised on: 07/26/06
  
  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/

#ifndef BOUNDS_H
#define BOUNDS_H

#include "dilate.h"
#include "utilities.h"

#define BND_IN			2
#define BND_OUT		0
#define BND_PART_ROW	10
#define BND_PART_COL	11
#define BND_PART_ALL	12

/*class MatBounds
  There are two main types of bound checking:
    The most-northwest-corner one (BND_CHK_start*) 
    and the most-southeast-corner one (BND_CHK_end*).
  All the *Check* functions defined in this class inline calls to the above
    depending on the section of matrix that we want to compute on.
  Any member with a "1" at the end of the name reffers to cases where the
    number of colunms in the matrix is = 1 + the number of rows. This is 
    particularly helpfull when an Rhs is appended to an original square 
    matrix, and the computations on the matrix need to be extended to the Rhs
    as well.
  the "d" at the beginning of member variables --> dilated integers.
  
member functions' parameters: 
  *bChk or *bndChk: Current bound checking state
      --> is the parent block currently fully within the bounds (BND_IN),
                or partially in rowwise(BND_PART_ROW), 
                or partially in columnwise(BND_PART_COL),
                or partially in either way(BND_PART_ALL),
                or fully outide of bounds (BND_OUT) ?
  mbIndex: morton Index of the current block.
  dRowBnd: dilated row index of the bound.
  dColBnd: dilated column index of the bound.
  level: level in tree.
*/
class MatBounds
{
private:
	indexType* bSize;  //pointer to precomputed block sizes based on tree level

	void BND_CHK_start(int* bChk, indexType mbIndex, 
	                   indexType dRowBnd, indexType dColBnd, int level);
	int BND_CHK_start_All( indexType mbIndex, 
	                       indexType dRowBnd, indexType dColBnd, int level);
	int BND_CHK_start_Row(indexType mbIndex, indexType dRowBnd, int level);
	int BND_CHK_start_Col(indexType mbIndex, indexType dColBnd, int level);
	void BND_CHK_end(int* bChk, indexType mbIndex, 
	                 indexType dRowBnd, indexType dColBnd, int level);
	int BND_CHK_end_All( indexType mbIndex, 
	                      indexType dRowBnd, indexType dColBnd, int level);
	int BND_CHK_end_Row(indexType mbIndex, indexType dRowBnd, int level);
	int BND_CHK_end_Col(indexType mbIndex, indexType dColBnd, int level);

public:
	int dStartRow, dStartCol;        //northwest corner 
	int dMidRow, dMidCol;            //current pivot point (if pivoting)
	int dEndRow, dEndCol, dEndCol1;  //southwest corner
	int dEndRow_HL, dEndCol_HL, dEndCol1_HL; //Higher bits of ending bounds
	int baseRows, baseCols, baseCols1; //lower [baseOrderLg] bits of ending 
	                                     //bounds.  

  //--------------- Constructors and Initializing ------------------------
	MatBounds():	dStartRow(0), dStartCol(0) {;}
	MatBounds(int rows, int cols, indexType* bS):	
                                dStartRow( 0 ),
																dStartCol( 0 ),
																dEndRow( evenDilate(rows) ),
																dEndCol( oddDilate(cols) ),
																dEndRow_HL( dEndRow / baseSize ),
																dEndCol_HL( dEndCol / baseSize ),
																baseRows( rows%baseOrder ),
																baseCols( cols%baseOrder ),
																bSize( bS )
																{}
	MatBounds(int rows, int cols, int cols1, indexType* bS):	
                                dStartRow( 0 ),
																dStartCol( 0 ),
																dEndRow( evenDilate(rows) ),
																dEndCol( oddDilate(cols) ),
																dEndCol1( oddDilate(cols1) ),
																dEndRow_HL( dEndRow / baseSize ),
																dEndCol_HL( dEndCol / baseSize ),
																dEndCol1_HL( dEndCol1 / baseSize ),
																baseRows( rows%baseOrder ),
																baseCols( cols%baseOrder ),
																baseCols1( cols1%baseOrder ),
																bSize( bS )
																{}
	void init(int rows, int cols, indexType* bS){
		dEndRow = evenDilate(rows);
		dEndCol = oddDilate(cols);

		dEndRow_HL = dEndRow / baseSize;
		dEndCol_HL = dEndCol / baseSize;

		baseRows = rows%baseOrder;
		baseCols = cols%baseOrder;
		bSize = bS;
	}
	void init(int rows, int cols, int cols1, indexType* bS){
		dEndRow = evenDilate(rows);
		dEndCol = oddDilate(cols);
		dEndCol1 = oddDilate(cols1);

		dEndRow_HL = dEndRow / baseSize;
		dEndCol_HL = dEndCol / baseSize;
		dEndCol1_HL = dEndCol1 / baseSize;

		baseRows = rows%baseOrder;
		baseCols = cols%baseOrder;
		baseCols1 = cols1%baseOrder;
		bSize = bS;
	}
	
	//-------------  Single-Bound Checking -------------------------
	
	__attribute__ ((always_inline))
	void startCheck(int* bChk, indexType mbIndex, int level) {
		BND_CHK_start(bChk, mbIndex, dStartRow, dStartCol, level);
	}
	__attribute__ ((always_inline))
	void nwCheck_end(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex, dMidRow, dMidCol, level);
	}
	__attribute__ ((always_inline))
	void swCheck_start(int* bChk, indexType mbIndex, int level) {
		BND_CHK_start(bChk, mbIndex, dMidRow, dStartCol, level);
	}
	__attribute__ ((always_inline))
	void swCheck_end(int* bChk, indexType mbIndex, int level) {
		BND_CHK_start(bChk, mbIndex, dEndRow, dMidCol, level);
	}
	__attribute__ ((always_inline))
	void neCheck_start(int* bChk, indexType mbIndex, int level) {
		BND_CHK_start(bChk, mbIndex, dStartRow, dMidCol, level);
	}
	__attribute__ ((always_inline))
	void neCheck_end(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex, dMidRow, dEndCol1, level);
	}
	__attribute__ ((always_inline))
	void seCheck_start(int* bChk, indexType mbIndex, int level) {
		BND_CHK_start(bChk, mbIndex, dMidRow, dMidCol, level);
	}
	__attribute__ ((always_inline))
	void seCheck_end(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex, dEndRow, dEndCol1, level);
	}
	__attribute__ ((always_inline))
	void endCheck0(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex, dEndRow, dEndCol, level);
	}
	__attribute__ ((always_inline))
	void endCheck(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex*bSize[level], dEndRow, dEndCol, level);
	}
	__attribute__ ((always_inline))
	void endCheck1(int* bChk, indexType mbIndex, int level) {
		BND_CHK_end(bChk, mbIndex*bSize[level], dEndRow, dEndCol1, level);
	}
	
	//-------------  Double-Bound Checking -------------------------
	// first, check most northwest corner. 
	// If success, check most southeast corner.
	
	__attribute__ ((always_inline))
	void check(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		startCheck(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			endCheck0(ebChk, mbIndex, level);
	}
	__attribute__ ((always_inline))
	void check1(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		startCheck(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			seCheck_end(ebChk, mbIndex, level);
	}
	__attribute__ ((always_inline))
	void nwCheck(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		startCheck(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			nwCheck_end(ebChk, mbIndex, level);
	}
	__attribute__ ((always_inline))
	void swCheck(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		swCheck_start(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			swCheck_end(ebChk, mbIndex, level);
	}
	__attribute__ ((always_inline))
	void neCheck(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		neCheck_start(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			neCheck_end(ebChk, mbIndex, level);
	}
	__attribute__ ((always_inline))
	void seCheck(int* sbChk, int* ebChk, indexType mbIndex, int level){
		mbIndex *= bSize[level];
		seCheck_start(sbChk, mbIndex, level);
		if(sbChk != BND_OUT)
			seCheck_end(ebChk, mbIndex, level);
	}
	
	//------ At base case, get the block's row & col limits ----------
	// return row limit in I or iMax, and col limit in I or jMax
	
	__attribute__ ((always_inline))
	void getBaseLimits(int sbndChk, int ebndChk, int* I, int* J){
		assert(sbndChk == BND_IN);
		getEndBaseLimits(ebndChk, I, J);
	}
	void getEndBaseLimits(int bndChk, int* iMax, int* jMax){
		if(bndChk == BND_IN){
			*iMax = *jMax = baseOrder;
		}
		else if(bndChk == BND_PART_ROW){
			*iMax = baseRows;
			*jMax = baseOrder;
		}
		else if(bndChk == BND_PART_COL){
			*iMax = baseOrder;
			*jMax = baseCols;
		}
		else{
			*iMax = baseRows;
			*jMax = baseCols;
		}
	}
	__attribute__ ((always_inline))
	void getBaseLimits1(int sbndChk, int ebndChk, int* I, int* J){
		assert(sbndChk == BND_IN);
		getEndBaseLimits1(ebndChk, I, J);
	}
	void getEndBaseLimits1(int bndChk, int* iMax, int* jMax){
		if(bndChk == BND_IN){
			*iMax = *jMax = baseOrder;
		}
		else if(bndChk == BND_PART_ROW){
			*iMax = baseRows;
			*jMax = baseOrder;
		}
		else if(bndChk == BND_PART_COL){
			*iMax = baseOrder;
			*jMax = baseCols1;
		}
		else{
			*iMax = baseRows;
			*jMax = baseCols1;
		}
	}

  //Destructor
	~MatBounds(){}
};

#endif
/////////////////////////////////////////////////////////////////////////////

