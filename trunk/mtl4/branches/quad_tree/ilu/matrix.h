/*****************************************************************************
  file: matrix.h
  --------------
  Defines the class Matrix

  Revised on: 07/25/06

  D. S. Wise
  Larisse D. Voufo
*****************************************************************************/
#ifndef MATRIX_H
#define MATRIX_H

#include "bounds.h"
#include "stock.h"


class Matrix
{
private:
	int rows, cols,  //dimensions
	     expRows, expCols, //dimensions expanded to next multiple of baseOrder
	     rows2;      // rows*2, for prefetching

	void insertTree( Both* node, indexType mbIndex,
					int row, int col, int nnext, dataType data, int rlevel);

	__attribute__ ((always_inline))
	inline void init() {
	 //set dimensions
		rows2 = 2*rows;
		expRows = expandSide(rows);
		expCols = expandSide(cols);

//printf("  %d  %d  -  %d  %d\n", rows, cols, expRows, expCols);
    //determine max numb of quads and base blocks ever needed
      //matrix's morton size
		indexType mSize = evenDilate(expRows-1)+oddDilate(expCols-1)+1;
//printf("  %u \n", mSize);
		  //max number of dense base blocks needed
		nDensebBlks = (expRows*expCols)/baseSize;
//printf("  %u  -  %u\n", nDensebBlks, nDensebBlks*2 );
		  //max number of leaf nodes
		  //if the matrix was to produce a complete quadtree
		indexType maxNbBlks = powerOf4(nextLog4(mSize/baseSize));
//printf("  %u \n", maxNbBlks);
		  //max number of Quad nodes needed
		maxNQuads = (maxNbBlks-1)/3 - (maxNbBlks - mSize/baseSize)/4;
//printf("  %u  -  %u\n", maxNQuads, maxNQuads*2 );

    //set level-related variables
		maxLevel = nextLog4(maxNbBlks);
//printf("  %d \n", maxLevel);
		maxLevel2 = maxLevel * 2;
		maxLevelBOL = maxLevel + baseOrderLg;

    //pre-compute subBlock sizes based on level in tree
		bSize = (indexType*)(valloc((maxLevel+1)*sizeof(indexType)));
		for(int i=0; i<=maxLevel; i++)
			bSize[i] = powerOf4((maxLevel-i))*baseSize;

    //initialize bounds
		bnd.init(rows, cols, cols+1, bSize);

		//increment the stocks allocation requirements
		// *2, since saving a copy of the original matrix in backupNode
		int nq = maxNQuads*2;
		int nb = nDensebBlks*2;
		if(nq < maxNQuads)
		  nq = std::numeric_limits<indexType>::max();
		if(nb < nDensebBlks)
		  nb = std::numeric_limits<indexType>::max();
		max_alloc_requirements(nq, nb);

		//initialize rhsInit()
		rhs = (dataType*)(valloc(rows*sizeof(dataType)));
		
		numb_fill_in = 0;
	}

public:
  int numb_fill_in;
	Both mainNode;       // head node of the matrix
	Both backupNode;     // head node for the backup of the matrix
	dataType* rhs;       // right hand side of the system of equations
	indexType* bSize;    // pre-computed sub-block sizes
	int maxLevel,        //max numb of levels in tree,
	     maxLevel2, maxLevelBOL;  //maxLevel*2, maxLevel+baseOrderLg
	                                 //--> prefetching
	int symmetric;       // when Harwell Boeing input, is it symmetric?
	MatBounds bnd;       // matrix bounds
	indexType maxNQuads, nDensebBlks;  // max numb of quadNodes and base blocks

  //constructors and initializations
	Matrix() :	mainNode(NIL), backupNode(NIL), rhs(NIL)
			{;}
	Matrix(int r, int c):	rows(r), cols(c),
							mainNode(NIL), backupNode(NIL), rhs(NIL)
			{ init(); }
	void init(int r, int c){	rows = r; cols = c; symmetric = 0; init(); 	}
	void init(int r, int c, int sym)
	         {	rows = r; cols = c; symmetric = sym; init(); 	}

	//return rows, rows2, cols
	int getRows() { return rows; }
	int getRows2() { return rows2; }
	int getCols() { return cols; }

	//building tree
	void buildTree_HB();
	void buildTree_RM();

	//copyMatrix
	void copyMatrix(Both* to, Both from, int level);

	//printing tree
	void printTree(Both node, indexType mbIndex, int rlevel);
	void printRowMaj(Both node);
	void printRow(Both node, int rlevel, int index, int indexBase);

  void freeNodes(Both* node, int level);
  
	//re-initialization
	void kill()
	{
		mainNode = NIL;
		backupNode = NIL;
		free(rhs);
	}

	//destructor
	~Matrix(){}
};

extern Matrix mat;

#endif
/////////////////////////////////////////////////////////////////////////////

