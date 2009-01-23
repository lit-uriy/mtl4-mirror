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

	template <typename Vector, typename Functor>
	inline void for_each_nonzero(const Vector& v, Functor& f) 
	{
	    typedef typename traits::range_generator<tag::nz, Vector>::type cursor;
	    for (cursor i= begin<tag::nz>(v), iend= end<tag::nz>(v); i != iend; ++i)
		f(i);
	}
    }

    namespace matrix {

	template <typename Matrix, typename Functor>
	inline void for_each_nonzero(const Matrix& m, Functor& f)
	{
	    typedef typename mtl::traits::range_generator<tag::major, Matrix>::type     cursor_type;
	    typedef typename mtl::traits::range_generator<tag::nz, cursor_type>::type   icursor_type;
	    
	    for (cursor_type cursor = begin<tag::major>(m), cend = end<tag::major>(m); cursor != cend; ++cursor) 
		for (icursor_type icursor = begin<tag::nz>(cursor), icend = end<tag::nz>(cursor); 
		     icursor != icend; ++icursor)
		    f(icursor);
	}
    }


} // namespace mtl

#endif // MTL_FOR_EACH_NONZERO_INCLUDE
