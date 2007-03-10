// $COPYRIGHT$

#ifndef MTL_COPY_INCLUDE
#define MTL_COPY_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/operation/set_to_zero.hpp>

#include <iostream>
#include <boost/numeric/mtl/operation/print.hpp>

namespace mtl {

    namespace matrix {
	
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

	template <typename MatrixSrc, typename MatrixDest>
	inline void copy(const MatrixSrc& src, MatrixDest& dest)
	{
	    dest.change_dim(src.num_rows(), src.num_cols());
	    detail::zero_with_sparse_src(dest, typename traits::category<MatrixSrc>::type());

	    typename traits::row<MatrixSrc>::type             row(src); 
	    typename traits::col<MatrixSrc>::type             col(src); 
	    typename traits::const_value<MatrixSrc>::type     value(src); 
	    typedef typename traits::range_generator<tag::major, MatrixSrc>::type  cursor_type;
	    
	    matrix_inserter<MatrixDest>   ins(dest);
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

    } // namespace matrix

#if 0
    template <typename MatrixSrc, typename MatrixDest>
    inline void copy(const MatrixSrc& src, tag::matrix, MatrixDest& dest, tag::matrix)
    // inline void copy(const MatrixSrc& src, tag::matrix_expr, MatrixDest& dest, tag::matrix)
    {
	return matrix::copy(src, dest);
    }


    template <typename CollSrc, typename CollDest>
    inline void copy(const CollSrc& src, CollDest& dest)
    {
	return copy(src, traits::category<CollSrc>::type(), dest, traits::category<CollDest>::type());
    }
#endif

} // namespace mtl

#endif // MTL_COPY_INCLUDE
