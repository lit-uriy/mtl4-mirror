/*////////////////////////////////////////////////////////////////////////////
 file matrix.cpp

 Sparse matrices representation - matrix I/O

 D. S. Wise
 Larisse D. Voufo
 12/15/2005

////////////////////////////////////////////////////////////////////////////*/

#include "matrix.h"

/*
void buildTree_HB()
  Build a quad-tree from a Harwell-Boeing input.
  Reads, from std in, and store the col pointers and row indices.
  Then, for each non-zero elt, read value from std in, and inssert in tree.
*/void Matrix::buildTree_HB()
{
	int i, j, si, bi, bj, nnzero;
	indexType mbIndex;
	dataType data;

	//read in column pointers into [colPtr]
	scanf("%d", &nnzero);
	int *colPtr = (int*)(valloc((cols+1)*sizeof(int)));
	for(i=0; i<cols+1; i++) {
		scanf("%d", &colPtr[i]);
	}
//printf("%d  %d\n", nnzero, colPtr[cols]);
	  
	// the number of values (non-zero) = the last element in colPtr - 1
	//nnzero = colPtr[cols]-1;

  //read in row indices into [rowInd]
	int *rowInd = (int*)(valloc(nnzero*sizeof(int)));
	for(i=0; i<nnzero; i++)
		scanf("%d", &rowInd[i]);

//printf("  Here \n");
	//for each column
	for (j=0; j<cols; j++) {
		//for each row in col
		for(si=colPtr[j]-1; si<colPtr[j+1]-1; si++) {
//printf("  Here A  %d\n", si);
			//We Have a Dense Block, let's insert the data!!!
			i = rowInd[si]-1;

//printf("  Here A  %d  %d\n", j, i);
			//read in data
			scanf("%lf", &data);

//printf("  Here B  %lf\n", data);
			//insert data
			bi = i/baseOrder;
			bj = j/baseOrder;
			mbIndex = evenDilate(bi) + oddDilate(bj);
			insertTree( &mainNode, mbIndex,
						i%baseOrder, j%baseOrder, 0, data, maxLevel2 );

			//if symmetric and i!=j, insert symmetric data as well
			if(symmetric && (i != j)) {
				if(bi != bj ) {
					mbIndex = evenDilate(j/baseOrder) + oddDilate(i/baseOrder);
				}
				insertTree( &mainNode, mbIndex,
							j%baseOrder, i%baseOrder, 0, data, maxLevel2 );
			}
		} // end of each row
	} // end of each column

//printf("  Here 2\n");
	// return allocated memory
	free(colPtr);
	free(rowInd);
}
/*
void buildTree_RM()
  Build a quad-tree from a Row Major input.
  Reads, from std in, and insert non-zero data into tree
*/
void Matrix::buildTree_RM()
{
	int i, j, bi, bj, t;
	indexType mbIndex;
  dataType data;
  
	//for each row
	for(i = 0; i< cols; i++) {
		//for each col
		for(j=0; j<cols; j++) {

			//read in data
			scanf("%lf", &data);

			//if data is non-zero, insert remaining data up to the next
			//	multiple of [baseOrder] col.
			if(data != 0.0) {
				//number od data to read in afterwards
				t = min( expandNextSide(j), cols) - j - 1;
				//insert data(s)
				bi = i/baseOrder;
				bj = j/baseOrder;
				mbIndex = evenDilate(bi) + oddDilate(bj);
				insertTree( &mainNode, mbIndex,
							i%baseOrder, j%baseOrder, t, data, maxLevel2 );
				j += t;	//move to next multiple of [baseOrder] col index
			}
		}// End - for each col
	} //End - for each row
}


/*
void insertTree( Both* node, indexType mbIndex,
					int row, int col, int nnext, dataType data, int rlevel)
  Insert data into a tree. within the same col in a base block,
  	read and add more data if specified.
  param
    node: head of the tree
    mbIndex: morton order of the block to perform insertion on
    row:	row index within base block;
    col: 	col index within base block;
    nnext:	number of data to read and add later.
    rlevel: remaining number of levels in tree.
            starts at maxLevel*2 and keep decrementing by 2 untill riching 0.
            *2 since the bit mask needed for determining the flow of control.
*/
void Matrix::insertTree( Both* node, indexType mbIndex,
					int row, int col, int nnext, dataType data, int rlevel)
{
	if(rlevel>0){  // if node is quadNode
		if(*node==NIL)
			*node = (Both)fromQuadNode();
		rlevel -= 2;
		//which quadrant,of the current block, does the dense block belong to?
		int bit = (mbIndex >> rlevel) & 3;
		switch(bit) {
			case 0:	//NW
				insertTree( &(NW(*node)), mbIndex, row, col, nnext, data, rlevel );
				break;
			case 1:	//SW
				insertTree( &(SW(*node)), mbIndex, row, col, nnext, data, rlevel );
				break;
			case 2:	//NE
				insertTree( &(NE(*node)), mbIndex, row, col, nnext, data, rlevel );
				break;
			case 3:	//SE
				insertTree( &(SE(*node)), mbIndex, row, col, nnext, data, rlevel );
				break;
		}
	}
	else { // if node is baseBlock
		if(*node==NIL)
			*node = (Both)fromBaseBlock();
		//insert data
		int i, j, t;
#if MTN_BBLK
		i = evenDilate(row);
		j = oddDilate(col);
#else
		i = row*baseOrder;
		j = col;
#endif
		((baseBlock)(*node))[i+j] = data;

		//read and insert more data
#if MTN_BBLK
		for( t=0, oddInc(j); t<nnext; t++, oddInc(j) ) {
#else
		for( t=0, ++j; t<nnext; t++, j++ ) {
#endif
			scanf("%lf", &data);
			((baseBlock)(*node))[i+j] = data;
		}
	}
}

/*
void copyMatrix(Both* to, Both from, int level)
  make a copy of tree representation of a matrix
  param
    to: node to coy into
    from: node to copy from
    level: level in tree. starts at LEVEL_START (0)
*/
void Matrix::copyMatrix(Both* to, Both from, int level)
{
	if(from == NIL){
	}
	else if (level<maxLevel) { // if from is a quadNode
		*to = (Both)fromQuadNode();
		level++;  //next level
		copyMatrix( &(NW(*to)), NW(from), level);
		copyMatrix( &(SW(*to)), SW(from), level);
		copyMatrix( &(NE(*to)), NE(from), level);
		copyMatrix( &(SE(*to)), SE(from), level);
	}
	else {   //if from is a base block
		int i;
		*to = (Both)fromBaseBlock();
		for(i=0; i<baseSize; i++)
			((baseBlock)(*to))[i] = ((baseBlock)from)[i] ;
	}
}

/*
void printTree(Both node, indexType mbIndex, int level)
  Print a tree.
  param
    node: head node of tree to print
    mbIndex: current morton block index. starts at MTN_START (0)
    level: level in tree. starts at LEVEL_START (0)
*/
void Matrix::printTree(Both node, indexType mbIndex, int level)
{
	if(node == NIL){
	  printTabs(level);
		printf("()\n");
	}
	else if (level < maxLevel) { // if node is a quadNode
	  printTabs(level);
		printf("(\n");
		mbIndex *= 4;
		level++;
		printTree(NW(node), mbIndex, level);    //NW
		printTree(SW(node), mbIndex+1, level);	//SW
		printTree(NE(node), mbIndex+2, level);	//NE
		printTree(SE(node), mbIndex+3, level);	//SE
	  printTabs(level);
		printf(")\n");
	}
	else { // if node is base block
		int i, j, iMax, jMax;
	  printTabs(level);
		printf("( \n");
#if MTN_BBLK
		iMax = evenDilate( ((mbIndex & evenBits) == bnd.dEndRow_HL) ?
										bnd.baseRows: baseOrder );
		jMax = oddDilate( ((mbIndex & oddBits) == bnd.dEndCol_HL) ?
										bnd.baseCols : baseOrder );
		for( i=0; i<iMax; evenInc(i) ) {
			printTabs(level);
			for( j=0; j<jMax; oddInc(j) ) {
#else
		iMax = ( ((mbIndex & evenBits) == bnd.dEndRow_HL) ?
										bnd.baseRows*baseOrder : baseSize);
		jMax = ( ((mbIndex & oddBits) == bnd.dEndCol_HL) ?
										bnd.baseCols : baseOrder );
		for(i=0; i<iMax; i+=baseOrder) {
			printTabs(level);
			for(j=0; j<jMax; j++) {
#endif
				printf( "\t%lf", ((baseBlock)node)[i+j] );
			}
			printf("\n");
		}
	  printTabs(level);
		printf(")\n");
	}
}

/*
void printRowMaj(Both node)
  Print the row major representation of a matrix represented by a quadtree.
  param
    node: head node of tree to print in row-major
*/
void Matrix::printRowMaj(Both node)
{
	int inputH, inputB, N=rows/baseOrder, Nb=bnd.baseRows;
  	int j;

  	for(inputH=0; inputH<N; inputH++){
		for(inputB=0; inputB<baseOrder; inputB++){
			//Print row
			gRowIndex = 0;
			printf("\n");
			printRow(node, maxLevel, inputH, inputB);
		}
		printf("\n");
  	}
	for(inputB=0; inputB<Nb; inputB++){
		//Print row
		gRowIndex = 0;
		printf("\n");
		printRow(node, maxLevel, N, inputB);
	}
	printf("\n");
}


void Matrix::freeNodes(Both* node, int level){
  if(*node==NIL){
  }
  else if (level<maxLevel) {
    level++;
    freeNodes(&(NW(*node)), level);
    freeNodes(&(SW(*node)), level);
    freeNodes(&(NE(*node)), level);
    freeNodes(&(SE(*node)), level);
    toQuadNode((quadNode**)(node));
  }
  else {
    toBaseBlock((baseBlock*)(node));
  }
}


/*
void printRow(Both node, int rlevel, int index, int indexBase)
  Print a row from the quadtree representation of a matrix.
  param
    node: head node of tree to print the row from
    rlevel: remaining levels in tree. starts at [maxLevel]
    index: higher bits of the row index to print
    indexbase: [baseOrderLog] lower bits of the row index to print
*/
void Matrix::printRow(Both node, int rlevel, int index, int indexBase)
{
	if(node == NIL) {
		double zero = 0;
		int t = min( powerOf2(rlevel+baseOrderLg), cols-gRowIndex );
		for(int j=0; j<t; j++) {
			printf("%lf\t", zero);
		}
		gRowIndex += t;
	}
	else if(rlevel>0) {  //if node is a quad node
		rlevel--;
		int bit = (index >> rlevel) & 1;
		if(!bit) {
			printRow(NW(node), rlevel, index, indexBase);
			printRow(NE(node), rlevel, index, indexBase);
		}
		else{
			printRow(SW(node), rlevel, index, indexBase);
			printRow(SE(node), rlevel, index, indexBase);
		}
	}
	else  {  //if node is base block
		int i, j, jmax;
#if MTN_BBLK
		i = evenDilate(indexBase);
		jmax = oddDilate( min(baseOrder, cols - gRowIndex) );
		for( j=0; j<jmax; oddInc(j) ) {
#else
		i = indexBase*baseOrder;
		jmax = min(baseOrder, cols - gRowIndex);
		for( j=0; j<jmax; j++ ) {
#endif
			printf("%lf\t", ((baseBlock)node)[i+j] );
		}
		gRowIndex += jmax;
	}
}


/////////////////////////////////////////////////////////////////////////////
