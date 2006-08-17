
/* matrix environment setup for incomplete LU
   + iterative solvers
   pass in a *global* variable, matrix object mat
*/

#ifndef OP_ENV_SETUP_H
#define OP_ENV_SETUP_H

#include "matrix.h"
#include "defs.h"

static void buildTree(char* fmt)
{
	if(strcmp(fmt, ROW_MAJ_Z)==0){
		mat.buildTree_RM();
	}
	else if( (strcmp(fmt, HB)==0) || (strcmp(fmt, HBS)==0) ){
		mat.buildTree_HB();
	}
	else{
		printf("Invalid Input Format:  %s\n",fmt);
		exit(-17);
	}
}

static void setup_env(Matrix& mat)
{
  cout<<"Here***\n";
  int nRows, nCols, sym, numInputMat;
  char fmt[MAX_FLAG_SIZE];

  //computer base order and size - related variables
  baseOrderLg = 3;
  baseOrder = powerOf2(baseOrderLg);
  baseSize = powerOf4(baseOrderLg);
  baseOrder2 = baseOrder * 2;
  baseSize2 = baseSize *2;

  scanf("%d",&numInputMat);
  assert (numInputMat == 1);

  //get nRows, nCols, and fmt, and set sym if needed.
  scanf("%d", &nRows);
  scanf("%d", &nCols);
  scanf("%s", fmt);
  if(strcmp (fmt, HBS) == 0)
    sym = 1;

  //create the stocks for baseBlocks and quadNodes
  // --> determines how many "data" (quadNodes or baseBlocks)
  //to allocate at a time
  // --> sets the stepping into the memory space allocated for base blocks
  createStocks();

  //define Matrix - will set the maximum allocation requirement,
  // which is the number of "data" that will ever be needed for the given
  // matrix.
  mat.init(nRows, nCols, sym);

  // initialize the stocks, by calling initMem() for each stock type.
  initStocks();

  //Get Data and build the quadtree
  numb_bblocks = 0;
  cout<< nRows<<"\t"<<nCols<<endl;
  buildTree(fmt);

  // in case we are using the array of dataTypes, rhs,
  //defined as a member of Matrix.
  //readRhs();
}

#endif // OP_ENV_SETUP_H

//////////////////////////////////////////////////////////////////////////////
