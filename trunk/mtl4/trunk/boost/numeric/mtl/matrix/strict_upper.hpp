// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_STRICT_UPPER_INCLUDE
#define MTL_MATRIX_STRICT_UPPER_INCLUDE

namespace mtl { namespace matrix {

namespace traits {

    template <typename Matrix>
    struct strict_upper
    {
	typedef typename traits::bands<Matrix>::type type;
    };
}

template <typename Matrix> 
typename traits::strict_upper<Matrix>::type
inline strict_upper(const Matrix& A)
{
    return bands(A, 1, std::numeric_limits<long>::max());
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_STRICT_UPPER_INCLUDE
