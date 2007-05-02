// $COPYRIGHT$

#ifndef MTL_COPY_INCLUDE
#define MTL_COPY_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>
#include <boost/numeric/mtl/operation/update.hpp>

#include <iostream>
#include <boost/numeric/mtl/operation/print.hpp>

namespace mtl {
	
    namespace detail {

	// Set Destination matrix to zero when source is sparse 
	// (otherwise everything is overwritten anyway)
	template <typename MatrixDest>
	inline void zero_with_sparse_src(MatrixDest& dest, tag::sparse)
	{
	    set_to_zero(dest);
	}
	
	template <typename MatrixDest>
	inline void zero_with_sparse_src(MatrixDest& dest, tag::universe) {} 

    } // namespace detail


    template <typename Updater, typename MatrixSrc, typename MatrixDest>
    inline void gen_matrix_copy(const MatrixSrc& src, MatrixDest& dest, bool with_reset)
    {
	// Deprecated, will  be removed
	// dest.change_dim(num_rows(src), num_cols(src));
	MTL_THROW_IF(num_rows(src) != num_rows(dest) || num_cols(src) != num_cols(dest), bad_range);

	if (with_reset)
	    detail::zero_with_sparse_src(dest, typename traits::category<MatrixSrc>::type());
	
	typename traits::row<MatrixSrc>::type             row(src); 
	typename traits::col<MatrixSrc>::type             col(src); 
	typename traits::const_value<MatrixSrc>::type     value(src); 
	typedef typename traits::range_generator<tag::major, MatrixSrc>::type  cursor_type;
	
	matrix::inserter<MatrixDest, Updater>   ins(dest);
	for (cursor_type cursor = begin<tag::major>(src), cend = end<tag::major>(src); 
	     cursor != cend; ++cursor) {
	    // std::cout << dest << '\n';
	    
	    typedef typename traits::range_generator<tag::nz, cursor_type>::type icursor_type;
	    for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); 
		 icursor != icend; ++icursor) {
		//std::cout << "in " << row(*icursor) << ", " << col(*icursor) << " insert " << value(*icursor) << '\n';
		ins(row(*icursor), col(*icursor)) << value(*icursor); }
	}
    }
	    
    /// Copy matrix \p src into matrix \p dest
    template <typename MatrixSrc, typename MatrixDest>
    inline void matrix_copy(const MatrixSrc& src, MatrixDest& dest)
    {
	gen_matrix_copy< operations::update_store<typename MatrixDest::value_type> >(src, dest, true);
    }
    

    // Add matrix \p src to matrix \p dest in copy-like style
    template <typename MatrixSrc, typename MatrixDest>
    inline void matrix_copy_plus(const MatrixSrc& src, MatrixDest& dest)
    {
	gen_matrix_copy< operations::update_plus<typename MatrixDest::value_type> >(src, dest, false);
    }
	
    // Subtract matrix \p src from matrix \p dest in copy-like style
    template <typename MatrixSrc, typename MatrixDest>
    inline void matrix_copy_minus(const MatrixSrc& src, MatrixDest& dest)
    {
	gen_matrix_copy< operations::update_minus<typename MatrixDest::value_type> >(src, dest, false);
    }
	

       
    template <typename MatrixSrc, typename MatrixDest>
    inline void copy(const MatrixSrc& src, tag::matrix, MatrixDest& dest, tag::matrix)
	// inline void copy(const MatrixSrc& src, tag::matrix_expr, MatrixDest& dest, tag::matrix)
    {
	return matrix_copy(src, dest);
    }


    template <typename CollSrc, typename CollDest>
    inline void copy(const CollSrc& src, CollDest& dest)
    {
	return copy(src, traits::category<CollSrc>::type(), dest, traits::category<CollDest>::type());
    }


} // namespace mtl

#endif // MTL_COPY_INCLUDE
