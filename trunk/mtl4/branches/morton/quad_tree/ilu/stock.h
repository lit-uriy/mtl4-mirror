/*****************************************************************************
  file: stock.h
  -------------- 
  Defines stocks, and functions that manipulate them.
  Two main stocks in use: 
    quadStock for quadNodes, and bBlockStock for baseBlocks
  Again, extern variables due to function inlining obligation. 

  Revised on: 07/25/06
  
  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef STOCK_H
#define STOCK_H

#include "utilities.h"

extern Memory<quadNode> quadStock;
extern Memory<dataType> bBlockStock;
extern int numb_bblocks;

/*
void compute_alloc_steps(int baseBlockByteSize)
  Compute de number of datatypes to allocate at a time. 
  To call ONCE for all storage types and matrices.
  param baseBlockByteSize: size in bytes of the base blocks
*/
__attribute__ ((always_inline))
inline void compute_alloc_steps(int baseBlockByteSize)
{
	quadStock.compute_alloc_steps(baseBlockByteSize);
}

/*
void createStocks()
  The stocks are already created during global variable instantiation
  However, still need to set the stock step for baseBlocks, as the value
  of baseSize is user dependant.
*/
__attribute__ ((always_inline))
inline void createStocks()
{
	compute_alloc_steps(baseSize*sizeof(dataType));
	bBlockStock.set_stock_step(baseSize);
}

/*
void max_alloc_requirements(int quad_inc, int bblk_inc)
  Increment the max number of quadNodes and baseBlocks ever needed.
  params
    quad_inc: incrementing value on quadNodes 
    bblk_inc: incrementing value on base blocks
*/
__attribute__ ((always_inline))
inline void max_alloc_requirements(indexType quad_inc, indexType bblk_inc)
{
	quadStock.max_alloc_requirement(quad_inc);
	bBlockStock.max_alloc_requirement(bblk_inc);
}

/*
void initStocks()
  Initialize of the stocks
*/
__attribute__ ((always_inline))
inline void initStocks()
{
	quadStock.initMem();
	bBlockStock.initMem();
}

/*
void destroyStocks()
  Destroy all the stocks
*/
__attribute__ ((always_inline))
inline void destroyStocks()
{
	quadStock.killMem();
	bBlockStock.killMem();
}

/*
void initQuadNode(quadNode* quad)
  initialize the given quadNode, setting its children to NIL
  param quad: pointer to quadNode to initialize 
*/
__attribute__ ((always_inline))
inline void initQuadNode(quadNode* quad)
{
	quad->NW = NIL;
	quad->SW = NIL;
	quad->NE = NIL;
	quad->SE = NIL;
}

/*
baseBlock fromBaseBlock()
  get a base block from bBlockStock.
  return pointer to dataType.
*/
__attribute__ ((always_inline))
inline baseBlock fromBaseBlock()
{
	numb_bblocks++;
	return bBlockStock.fromMem();
}

/*
void initBaseBlock(baseBlock* bBlk)
  Initialize the given base block, setting each data to 0 if it is NIL
  param bBlk: pointer to base block to initialize
*/
__attribute__ ((always_inline))
inline void initBaseBlock(baseBlock* bBlk)
{
	int t;
	if(*bBlk == NIL) {
		*bBlk = fromBaseBlock();
		for( t=0; t<baseSize; t++)
			(*bBlk)[t] == 0.0;
	}
}

/*
quadNode* fromQuadNode()
  get a quadNode from quadStock
  return a pointer to that mem address
*/
__attribute__ ((always_inline))
inline quadNode* fromQuadNode()
{
	quadNode* quad = quadStock.fromMem();
	initQuadNode(quad);
	return quad;
}

/*
void toQuadNode(quadNode* *quad)
  return a quadnode to quadStock
  param quad: quadNode to return
*/
__attribute__ ((always_inline))
inline void toQuadNode(quadNode* *quad)
{
	quadStock.toMem(quad);
}

/*
void toBaseBlock(baseBlock *bblk)
  return a base bllock to bBlockStock
  param bblk: baseBlock to return
*/
__attribute__ ((always_inline))
inline void toBaseBlock(baseBlock *bblk)
{
	bBlockStock.toMem(bblk);
}

/*
void normalizeQuad(quadNode** quad)
  Set current quadNode to NIL if its children are all NIL
  param quad: current quadNode
*/
__attribute__ ((always_inline))
inline void normalizeQuad(quadNode** quad)
{
	if(((*quad)->NW == NIL) && ((*quad)->SW == NIL) &&
		((*quad)->NE == NIL) && ((*quad)->SE == NIL) ) {
		toQuadNode(quad);
	}
}

/*
void normalizeBBlock(baseBlock* bBlk)
  Set current base Block to NIL if all data are = 0
  param bBlk: current base block
*/
__attribute__ ((always_inline))
inline void normalizeBBlock(baseBlock* bBlk)
{
	if( isNullArray(*bBlk, baseSize) ) {
		toBaseBlock(bBlk);
	}
}

#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////


