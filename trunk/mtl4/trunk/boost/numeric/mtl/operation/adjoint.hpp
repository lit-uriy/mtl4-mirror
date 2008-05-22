// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_ADJOINT_INCLUDE
#define MTL_ADJOINT_INCLUDE

#include <boost/numeric/mtl/operation/hermitian.hpp>

namespace mtl {

namespace traits {

    template <typename LinOp>
    struct adjoint
    {
	typedef matrix::hermitian_view<LinOp>  type;
	type operator()(const LinOp& A)
	{
	    return hermitian(A);
	}
    };
}

/// Adjoint linear operator, typically Hermitian transposed
template <typename LinOp>
typename traits::adjoint<LinOp>::type 
inline adjoint(const LinOp& A)
{
    return traits::adjoint<LinOp>()(A);
}

} // namespace mtl

#endif // MTL_ADJOINT_INCLUDE
