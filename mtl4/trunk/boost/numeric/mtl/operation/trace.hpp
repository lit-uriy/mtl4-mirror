// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_TRACE_INCLUDE
#define MTL_TRACE_INCLUDE

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace mtl {

template <typename Matrix>
typename Collection<Matrix>::value_type
inline trace(const Matrix& matrix)
{
    using math::one;
    typedef typename Collection<Matrix>::value_type value_type;

    throw_if(num_rows(matrix) != num_cols(matrix), matrix_not_square());

    // If matrix is empty then the result is the identity from the default-constructed value
    if (num_rows(matrix) == 0) {
	value_type ref;
	return one(ref);
    }

    value_type value= matrix[0][0];
    for (unsigned i= 1; i < num_rows(matrix); i++)
	value*= matrix[i][i];	
    return value;
}




} // namespace mtl

#endif // MTL_TRACE_INCLUDE
