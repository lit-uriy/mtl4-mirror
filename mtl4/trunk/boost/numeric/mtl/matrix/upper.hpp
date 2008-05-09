// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_UPPER_INCLUDE
#define MTL_MATRIX_UPPER_INCLUDE

#include <boost/numeric/mtl/matrix/bands.hpp>
#include <limits>

namespace mtl { namespace matrix {

namespace traits {

    template <typename Matrix>
    struct upper
    {
	typedef typename traits::bands<Matrix>::type type;
    };
}

template <typename Matrix> 
typename traits::upper<Matrix>::type
inline upper(const Matrix& A)
{
    return bands(A, 0, std::numeric_limits<long>::max());
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_UPPER_INCLUDE
