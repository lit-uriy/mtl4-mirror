// $COPYRIGHT$

#ifndef MTL_CHOLESKY_INCLUDE
#define MTL_CHOLESKY_INCLUDE

#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>


namespace mtl {

namespace with_bracket {

    // ============================================================================
    // Generic Cholesky factorization and operands for Cholesky on with submatrices
    // ============================================================================
	
    template < typename Matrix > 
    void cholesky_base (Matrix & matrix)
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
    void tri_solve_base(MatrixSW & SW, const MatrixNW & NW)
    {
	for (int k = 0; k < NW.num_rows (); k++) {
	    
	    for (int i = 0; i < SW.num_rows (); i++) {
		SW[i][k] /= NW[k][k];
		typename MatrixSW::value_type d = SW[i][k];
		
		for (int j = k + 1; j < SW.num_cols (); j++)
		    SW[i][j] -= d * NW[j][k];
	    }
	}
    }
    

    // Lower(SE) -= SW * SW^T
    template < typename MatrixSE, typename MatrixSW > 
    void tri_schur_base(MatrixSE & SE, const MatrixSW & SW)
    {
	for (int k = 0; k < SW.num_cols (); k++)
	    
	    for (int i = 0; i < SE.num_rows (); i++) {
		    typename MatrixSW::value_type d = SW[i][k];
		    for (int j = 0; j <= i; j++)
			SE[i][j] -= d * SW[j][k];
	    }
    }


    template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
    void schur_update_base(MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
    {
	for (int k = 0; k < NW.num_rows (); k++) 
	    for (int i = 0; i < NE.num_rows (); i++) {
		typename MatrixNW::value_type d = NW[i][k];
		for (int j = 0; j < NE.num_cols (); j++)
		    NE[i][j] -= d * SW[j][k];
	    }
    }


    // ======================
    // Corresponding functors
    // ======================
    
    struct cholesky_base_t
    {
	template < typename Matrix > 
	void operator() (Matrix & matrix)
	{
	    cholesky_base(matrix);
	}
    };
    
    struct tri_solve_base_t
    {
	template < typename MatrixSW, typename MatrixNW > 
	void operator() (MatrixSW & SW, const MatrixNW & NW)
	{
	    tri_solve_base(SW, NW);
	}
    };

    struct tri_schur_base_t
    {
	template < typename MatrixSE, typename MatrixSW > 
	void operator() (MatrixSE & SE, const MatrixSW & SW)
	{
	    tri_schur_base(SE, SW);
	}
    };

    struct schur_update_base_t
    {
	template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
	void operator() (MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
	{
	    schur_update_base(NE, NW, SW);
	}
    };

} // namespace with_bracket


// ==================================
// Functor types for Cholesky visitor
// ==================================


template <typename CholeskyBase, typename TriSolveBase, typename TriSchur, typename SchurUpdate,
	  typename BaseTest>
struct recursive_cholesky_visitor_t
{
    typedef  BaseTest                   base_test;

    template < typename Recurator > 
    bool is_base(const Recurator& recurator) const
    {
	return base_test()(recurator);
    }

    template < typename Matrix > 
    void cholesky_base(Matrix & matrix) const
    {
	CholeskyBase()(matrix);
    }

    template < typename MatrixSW, typename MatrixNW > 
    void tri_solve_base(MatrixSW & SW, const MatrixNW & NW) const
    {
	TriSolveBase()(SW, NW);
    }

    template < typename MatrixSE, typename MatrixSW > 
    void tri_schur_base(MatrixSE & SE, const MatrixSW & SW) const
    {
	TriSchur()(SE, SW);
    }

    template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
    void schur_update_base(MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW) const
    {
	SchurUpdate()(NE, NW, SW);
    }
};


namespace with_bracket {
    typedef recursive_cholesky_visitor_t<cholesky_base_t, tri_solve_base_t, tri_schur_base_t, schur_update_base_t,
					 recursion::bound_test_static<64> > 
               recursive_cholesky_base_visitor_t;
}


typedef with_bracket::recursive_cholesky_base_visitor_t                    recursive_cholesky_default_visitor_t;






namespace with_recurator {

    template <typename Recurator, typename Visitor>
    void schur_update(Recurator E, Recurator W, Recurator N, Visitor vis= Visitor())
    {
	if (E.is_empty() || W.is_empty() || N.is_empty())
	    return;

	if(vis.is_base(E)) {
	    typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value()),base_N(N.get_value());
	    vis.schur_update_base(base_E, base_W, base_N);
	} else{
	    schur_update(     E.north_east(),W.north_west()     ,N.south_west()     , vis);
	    schur_update(     E.north_east(),     W.north_east(),     N.south_east(), vis);
    
	    schur_update(E.north_west()     ,     W.north_east(),     N.north_east(), vis);
	    schur_update(E.north_west()     ,W.north_west()     ,N.north_west()     , vis);
    
	    schur_update(E.south_west()     ,W.south_west()     ,N.north_west()     , vis);
	    schur_update(E.south_west()     ,     W.south_east(),     N.north_east(), vis);
    
	    schur_update(     E.south_east(),     W.south_east(),     N.south_east(), vis);
	    schur_update(     E.south_east(),W.south_west()     ,N.south_west()     , vis);
	}
    }


    template <typename Recurator, typename Visitor>
    void tri_solve(Recurator S, Recurator N, Visitor vis= Visitor())
    {
        if (S.is_empty())
	    return;

        if(vis.is_base(S)) {   
	    typename Recurator::matrix_type  base_S(S.get_value()), base_N(N.get_value());
	    vis.tri_solve_base(base_S, base_N);
        } else{
     
	    tri_solve(S.north_west()     ,N.north_west(), vis);
	    schur_update(  S.north_east(),S.north_west()     ,N.south_west(), vis);
	    tri_solve(     S.north_east(),     N.south_east(), vis);
	    tri_solve(S.south_west()     ,N.north_west()     , vis);
	    schur_update(  S.south_east(),S.south_west()     ,N.south_west(), vis);
	    tri_solve(     S.south_east(),     N.south_east(), vis);
	}
    }


    template <typename Recurator, typename Visitor>
    void tri_schur(Recurator E, Recurator W, Visitor vis= Visitor())
    { 
        if (E.is_empty() || W.is_empty())
	    return;

        if (vis.is_base(W)) {
	    typename Recurator::matrix_type  base_E(E.get_value()), base_W(W.get_value());
	    vis.tri_schur_base(base_E, base_W);
        } else{ 
         
	    schur_update(E.south_west(),     W.south_west(),    W.north_west(), vis);
	    schur_update(E.south_west(),     W.south_east(),    W.north_east(), vis);
	    tri_schur(   E.south_east()     ,     W.south_east(), vis);
	    tri_schur(   E.south_east()     ,W.south_west()     , vis);
	    tri_schur(        E.north_west(),     W.north_east(), vis);
	    tri_schur(        E.north_west(),W.north_west()     , vis);
        }
    }


    template <typename Recurator, typename Visitor>
    void cholesky(Recurator recurator, Visitor vis= Visitor())
    {
        if (recurator.is_empty())
	    return;

        if (vis.is_base (recurator)){    
	    typename Recurator::matrix_type  base_matrix(recurator.get_value());
	    vis.cholesky_base (base_matrix);      
        } else {
	    cholesky(recurator.north_west(), vis);
	    tri_solve(    recurator.south_west(),       recurator.north_west(), vis);
	    tri_schur(    recurator.south_east(), recurator.south_west(), vis);
	    cholesky(     recurator.south_east(), vis);
        }
    }
        
} // namespace with_recurator



template <typename Backup= with_bracket::cholesky_base_t>
struct recursive_cholesky_t
{
    template <typename Matrix>
    void operator()(Matrix& matrix)
    {
	(*this)(matrix, recursive_cholesky_default_visitor_t());
    }

    template <typename Matrix, typename Visitor>
    void operator()(Matrix& matrix, Visitor vis)
    {
	apply(matrix, vis, typename traits::matrix_category<Matrix>::type());
    }   
 
private:
    // If the matrix is not sub-dividable then take backup function
    template <typename Matrix, typename Visitor>
    void apply(Matrix& matrix, Visitor, tag::universe)
    {
	Backup()(matrix);
    }

    // Only if matrix is sub-dividable, otherwise backup
    template <typename Matrix, typename Visitor>
    void apply(Matrix& matrix, Visitor vis, tag::qsub_dividable)
    {
	matrix_recurator<Matrix>  recurator(matrix);
	with_recurator::cholesky(recurator, vis);
    }
};


template <typename Matrix, typename Visitor>
inline void recursive_cholesky(Matrix& matrix, Visitor vis)
{
    recursive_cholesky_t<>()(matrix, vis);
}

template <typename Matrix>
inline void recursive_cholesky(Matrix& matrix)
{
    recursive_cholesky(matrix, recursive_cholesky_default_visitor_t());
}





template <typename Matrix>
void fill_matrix_for_cholesky(Matrix& matrix)
{
    typename Matrix::value_type   x= 1.0; 

    for (int i=0; i<matrix.num_rows(); i++) 
       for (int j=0; j<=i; j++)
	   if (i != j) {
	       matrix[i][j]= x; matrix[j][i]= x; 
	       x=x+1.0; 
	   }
  
    typename Matrix::value_type    rowsum;
    for (int i=0; i < matrix.num_rows(); i++) {
	rowsum= 0.0;
	for (int j=0; j<matrix.num_cols(); j++)
	    if (i!=j)
		rowsum += matrix[i][j]; 
	matrix[i][i]=rowsum*2;
    }       
}





} // namespace mtl

#endif // MTL_CHOLESKY_INCLUDE
