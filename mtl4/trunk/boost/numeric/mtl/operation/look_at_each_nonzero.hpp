// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_FOR_EACH_NONZERO_INCLUDE
#define MTL_FOR_EACH_NONZERO_INCLUDE

#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>


namespace mtl {

    namespace vector {

	/// Perform \p f(i) on each non-zero i in constant vector \p v; thus the must keep the result in its state 
	template <typename Vector, typename Functor>
	inline void look_at_each_nonzero(const Vector& v, Functor& f)
	{
	    typedef typename traits::range_generator<tag::iter::nz, Vector>::type iterator;
	    for (iterator i= begin<tag::iter::nz>(v), iend= end<tag::iter::nz>(v); i != iend; ++i)
		f(*i);
	}

    } // namespace vector

    namespace matrix {

	/// Perform a potentially mutating \p f(i) on each non-zero i in matrix \p A 
	template <typename Matrix, typename Functor>
	inline void look_at_each_nonzero(const Matrix& A, Functor& f)
	{
	    typename traits::value<Matrix>::type     value(A); 

	    typedef typename traits::range_generator<tag::major, Matrix>::type     cursor_type;
	    typedef typename traits::range_generator<tag::nz, cursor_type>::type   icursor_type;

	    for (cursor_type cursor = begin<tag::major>(A), cend = end<tag::major>(A); cursor != cend; ++cursor) 
		for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); 
		     icursor != icend; ++icursor)
		    f(value(*icursor));
	}

    } // namespace matrix

} // namespace mtl

#endif // MTL_FOR_EACH_NONZERO_INCLUDE
