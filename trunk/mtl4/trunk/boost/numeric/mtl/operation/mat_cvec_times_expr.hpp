// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MAT_CVEC_TIMES_EXPR_INCLUDE
#define MTL_MAT_CVEC_TIMES_EXPR_INCLUDE

#include <boost/numeric/mtl/operation/bin_op_expr.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl {

template <typename E1, typename E2>
struct mat_cvec_times_expr 
  : public bin_op_expr< E1, E2 >,
    public mtl::vector::vec_expr< mat_cvec_times_expr<E1, E2> >
{
    typedef bin_op_expr< E1, E2 >   base;

    // This is only a temporary solution, it is wrong when the matrix
    // is complex and the vector real -> otherwise more complicated
    typedef typename E2::value_type   value_type;
    typedef typename E2::size_type    size_type;
  
    mat_cvec_times_expr( E1 const& matrix, E2 const& vector )
	: base(matrix, vector)
    {}
};

} // namespace mtl

#endif // MTL_MAT_CVEC_TIMES_EXPR_INCLUDE
