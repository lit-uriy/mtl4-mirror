// $COPYRIGHT$

#ifndef MTL_CHOLESKY_INCLUDE
#define MTL_CHOLESKY_INCLUDE

namespace mtl {


// ============================================================================
// Generic Cholesky factorization and operands for Cholesky on with submatrices
// ============================================================================



template < typename Matrix > 
void cholesky (Matrix & matrix)
{
    for (int k = 0; k < matrix.num_rows (); k++) {
	matrix[k][k] = sqrt (matrix[k][k]);

	for (int i = k + 1; i < matrix.num_rows (); i++) {
	    matrix[i][k] /= matrix[k][k];
	    typename Matrix::value_type d = matrix[i][k];

	    for (int j = k + 1; j <= i; j++)
		matrix[i][j] -= d * matrix[j][k];
	}
    }
}


template < typename MatrixSW, typename MatrixNW > 
void tri_solve(MatrixSW & SW, const MatrixNW & NW)
{
  for (int k = 0; k < NW.num_rows (); k++) {

      for (int i = 0; i < SW.num_rows (); i++) {
	  SW[i][k] /= NW[k][k];
	  typename Matrix::value_type d = SW[i][k];

	  for (int j = k + 1; j < SW.num_cols (); j++)
	      SW[i][j] -= d * NW[j][k];
	}
    }
}


// Lower(SE) -= SW * SW^T
template < typename MatrixSE, typename MatrixSW > 
void tri_schur(MatrixSE & SE, const MatrixSW & SW)
{
    for (int k = 0; k < SW.num_cols (); k++)
	
	for (int i = 0; i < SE.num_rows (); i++) {
	    typename MatrixSW::value_type d = SW[i][k];
	    for (int j = 0; j <= i; j++)
		SE[i][j] -= d * SW[j][k];
	}
}


template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
void schur_update(MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
{
  for (int k = 0; k < NW.num_rows (); k++) 
      for (int i = 0; i < NE.num_rows (); i++) {
	  typename Matrix::value_type d = NW[i][k];
	  for (int j = 0; j < NE.num_cols (); j++)
	      NE[i][j] -= d * SW[j][k];
      }
}




struct cholesky_t
{
    template < typename Matrix > 
    void operator() (Matrix & matrix)
    {
	cholesky(matrix);
    }
};


struct tri_solve_t
{
    template < typename MatrixSW, typename MatrixNW > 
    void operator() (MatrixSW & SW, const MatrixNW & NW)
    {
	tri_solve(SW, NW);
    }
};


struct tri_schur_t
{
    template < typename MatrixSE, typename MatrixSW > 
    void operator() (MatrixSE & SE, const MatrixSW & SW)
    {
	tri_schur(SE, SW);
    }
};


struct schur_update_t
{
    template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
    void operator() (MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
    {

    }
};

} // namespace mtl

#endif // MTL_CHOLESKY_INCLUDE
