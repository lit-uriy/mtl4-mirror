
/* matrix environment setup for incomplete LU
   + iterative solvers
   pass in a *global* variable, matrix object mat
*/


FORCE_INLINE
inline void setup_env(Matrix& mat)
{
  int nRows, nCols;
  char fmt[MAX_FLAG_SIZE];

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
  buildTree(fmt);

  // in case we are using the array of dataTypes, rhs,
  //defined as a member of Matrix.
  //readRhs();
}

//////////////////////////////////////////////////////////////////////////////
