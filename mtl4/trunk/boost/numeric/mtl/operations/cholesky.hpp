// $COPYRIGHT$

#ifndef MTL_CHOLESKY_INCLUDE
#define MTL_CHOLESKY_INCLUDE

#include <boost/numeric/mtl/recursion/matrix_recurator.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>
#include <boost/numeric/mtl/operations/matrix_mult.hpp>
#include <boost/numeric/mtl/operations/assign_modes.hpp>
#include <boost/numeric/mtl/transposed_view.hpp>
#include <boost/numeric/mtl/recursion/base_case_cast.hpp>

namespace mtl {

namespace with_bracket {

    // ============================================================================
    // Generic Cholesky factorization and operands for Cholesky on with submatrices
    // ============================================================================
	
    template < typename Matrix > 
    void cholesky_base (Matrix & matrix)
    {
	for (int k = 0; k < matrix.num_cols(); k++) {
	    matrix[k][k] = sqrt (matrix[k][k]);
	    
		for (int i = k + 1; i < matrix.num_rows(); i++) {
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
	for (int k = 0; k < NW.num_cols (); k++) 
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


namespace with_iterator {

    // ============================================================================
    // Generic Cholesky factorization and operands for Cholesky on with submatrices
    // ============================================================================
    

    template < typename Matrix > 
    void cholesky_base (Matrix& matrix)
    {
	using namespace glas::tags; using traits::range_generator;

        typedef typename Matrix::value_type                         value_type;
        typedef typename range_generator<col_t, Matrix>::type       cur_type;             
        typedef typename range_generator<all_it, cur_type>::type    iter_type;            

	typedef typename range_generator<row_t, Matrix>::type       rcur_type;
	typedef typename range_generator<all_it, rcur_type>::type   riter_type;   
	
	typename Matrix::size_type k= 0;
	for (cur_type kb= begin<col_t>(matrix), kend= end<col_t>(matrix); kb != kend; ++kb, ++k) {

	    iter_type ib= begin<all_it>(kb), iend= end<all_it>(kb); 
	    ib+= k; // points now to matrix[k][k]

	    value_type root= sqrt (*ib);
	    *ib= root;

	    ++ib; // points now to matrix[k+1][k]
	    rcur_type rb= begin<row_t>(matrix); rb+= k+1; // to row k+1
	    for (int i= k + 1; ib != iend; ++ib, ++rb, ++i) {
		*ib = *ib / root;
		typename Matrix::value_type d = *ib;
		riter_type it1= begin<all_it>(rb);    it1+= k+1;      // matrix[i][k+1]
		riter_type it1end= begin<all_it>(rb); it1end+= i+1;   // matrix[i][i+1]
		iter_type it2= begin<all_it>(kb);     it2+= k+1;      // matrix[k+1][k]
		for (; it1 != it1end; ++it1, ++it2)
		    *it1 = *it1 - d * *it2;
	    }
	}
#if 0
	for (int k = 0; k < matrix.num_rows (); k++) {
	    matrix[k][k] = sqrt (matrix[k][k]);
	    
	    for (int i = k + 1; i < matrix.num_rows (); i++) {
		matrix[i][k] /= matrix[k][k];
		typename Matrix::value_type d = matrix[i][k];
		
		for (int j = k + 1; j <= i; j++)
		    matrix[i][j] -= d * matrix[j][k];
	    }
	}
#endif
    }

    
    template < typename MatrixSW, typename MatrixNW > 
    void tri_solve_base(MatrixSW & SW, const MatrixNW & NW)
    {
	using namespace glas::tags; using traits::range_generator;

        typedef typename range_generator<col_t, MatrixNW>::type       ccur_type;             
        typedef typename range_generator<all_cit, ccur_type>::type    citer_type;            

	typedef typename range_generator<row_t, MatrixSW>::type       rcur_type;
	typedef typename range_generator<all_it, rcur_type>::type     riter_type;   

	for (int k = 0; k < NW.num_rows (); k++) 
	    for (int i = 0; i < SW.num_rows (); i++) {

		typename MatrixSW::value_type d = SW[i][k] /= NW[k][k];

		rcur_type sw_i= begin<row_t>(SW);     sw_i+= i;  // row i
		riter_type it1= begin<all_it>(sw_i);  it1+= k+1; // SW[i][k+1]
		riter_type it1end= end<all_it>(sw_i);    
	
		ccur_type nw_k= begin<col_t>(NW);     nw_k+= k;  // column k
		citer_type it2= begin<all_cit>(nw_k); it2+= k+1; // NW[k+1][k]

		for(; it1 != it1end; ++it1, ++it2)
		    *it1 = *it1 - d * *it2;
	    }
    }
    

    // Lower(SE) -= SW * SW^T
    template < typename MatrixSE, typename MatrixSW > 
    void tri_schur_base(MatrixSE & SE, const MatrixSW & SW)
    {
	using namespace glas::tags; using traits::range_generator;

        typedef typename range_generator<col_t, MatrixSW>::type       ccur_type;             
        typedef typename range_generator<all_cit, ccur_type>::type    citer_type;            

	typedef typename range_generator<row_t, MatrixSE>::type       rcur_type;
	typedef typename range_generator<all_it, rcur_type>::type     riter_type;   

	for (int k = 0; k < SW.num_cols (); k++)
	    for (int i = 0; i < SE.num_rows (); i++) {
		typename MatrixSW::value_type d = SW[i][k];

		rcur_type se_i= begin<row_t>(SE);       se_i+= i;      // row i
		riter_type it1= begin<all_it>(se_i);                   // SE[i][0]
		riter_type it1end= begin<all_it>(se_i); it1end+= i+1;  // SE[i][i+i]

		ccur_type sw_k= begin<col_t>(SW);     sw_k+= k;        // column k
		citer_type it2= begin<all_cit>(sw_k);                  // SW[0][k]

		for(; it1 != it1end; ++it1, ++it2)
		    *it1 = *it1 - d * *it2;
	    }
    }


    template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
    void schur_update_base(MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
    {
	using namespace glas::tags; using traits::range_generator;

        typedef typename range_generator<col_t, MatrixSW>::type       ccur_type;             
        typedef typename range_generator<all_cit, ccur_type>::type    citer_type;            

	typedef typename range_generator<row_t, MatrixNE>::type       rcur_type;
	typedef typename range_generator<all_it, rcur_type>::type     riter_type;   

	for (int k = 0; k < NW.num_cols (); k++) 
	    for (int i = 0; i < NE.num_rows (); i++) {
		typename MatrixNW::value_type d = NW[i][k];

		rcur_type ne_i= begin<row_t>(NE);       ne_i+= i;      // row i
		riter_type it1= begin<all_it>(ne_i);                   // NE[i][0]
		riter_type it1end= end<all_it>(ne_i);                  // NE[i][num_col]

		ccur_type sw_k= begin<col_t>(SW);     sw_k+= k;        // column k
		citer_type it2= begin<all_cit>(sw_k);                  // SW[0][k]

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

} // namespace with_iterator


// ==================================
// Functor types for Cholesky visitor
// ==================================


template <typename BaseTest, typename CholeskyBase, typename TriSolveBase, typename TriSchur, typename SchurUpdate>
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


namespace detail {

    // Compute schur update with external multiplication; must have Assign == minus_mult_assign_t !!!
    template <typename MatrixMult>
    struct mult_schur_update_t
    {
	template < typename MatrixNE, typename MatrixNW, typename MatrixSW >
	void operator()(MatrixNE & NE, const MatrixNW & NW, const MatrixSW & SW)
	{
	    transposed_view<MatrixSW> trans_sw(const_cast<MatrixSW&>(SW)); 
	    MatrixMult()(NW, trans_sw, NE);
	}
    };

} // detail


namespace with_bracket {
    typedef recursive_cholesky_visitor_t<recursion::bound_test_static<64>, cholesky_base_t, tri_solve_base_t, 
					 tri_schur_base_t, schur_update_base_t > 
               recursive_cholesky_base_visitor_t;
}

namespace with_iterator {
    typedef recursive_cholesky_visitor_t<recursion::bound_test_static<64>, 
					 cholesky_base_t, tri_solve_base_t, tri_schur_base_t, schur_update_base_t>
               recursive_cholesky_base_visitor_t;
}

typedef with_bracket::recursive_cholesky_base_visitor_t                    recursive_cholesky_default_visitor_t;






namespace with_recurator {

    template <typename Recurator, typename Visitor>
    void schur_update(Recurator E, Recurator W, Recurator N, Visitor vis= Visitor())
    {
	using namespace recursion;

	if (E.is_empty() || W.is_empty() || N.is_empty())
	    return;

	if(vis.is_base(E)) {
	    typedef typename Visitor::base_test  base_test;
	    typedef typename base_case_matrix<typename Recurator::matrix_type, base_test>::type matrix_type;
	    
	    matrix_type  base_E(base_case_cast<base_test>(E.get_value())), 
		base_W(base_case_cast<base_test>(W.get_value())),
		base_N(base_case_cast<base_test>(N.get_value()));
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
	using namespace recursion;

        if (S.is_empty())
	    return;

        if(vis.is_base(S)) {   
	    typedef typename Visitor::base_test  base_test;
	    typedef typename base_case_matrix<typename Recurator::matrix_type, base_test>::type matrix_type;
	    
	    matrix_type  base_S(base_case_cast<base_test>(S.get_value())), 
		base_N(base_case_cast<base_test>(N.get_value()));

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
	using namespace recursion;

        if (E.is_empty() || W.is_empty())
	    return;

        if (vis.is_base(W)) {
	    typedef typename Visitor::base_test  base_test;
	    typedef typename base_case_matrix<typename Recurator::matrix_type, base_test>::type matrix_type;
	    
	    matrix_type  base_E(base_case_cast<base_test>(E.get_value())), 
               		 base_W(base_case_cast<base_test>(W.get_value()));
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
	using namespace recursion;

        if (recurator.is_empty())
	    return;

        if (vis.is_base (recurator)){    
	    typedef typename Visitor::base_test  base_test;
	    typedef typename base_case_matrix<typename Recurator::matrix_type, base_test>::type matrix_type;
	    
	    matrix_type  base_matrix(base_case_cast<base_test>(recurator.get_value()));
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
