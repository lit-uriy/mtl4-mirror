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

#ifndef MTL_HERMITIAN_INCLUDE
#define MTL_HERMITIAN_INCLUDE

#include <boost/numeric/mtl/matrix/hermitian_view.hpp>

namespace mtl { 

template <typename Matrix>
matrix::hermitian_view<Matrix> inline hermitian(const Matrix& matrix)
{
    return matrix::hermitian_view<Matrix>(matrix);
}


} // namespace mtl

#endif // MTL_HERMITIAN_INCLUDE
