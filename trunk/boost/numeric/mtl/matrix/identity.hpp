// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_IDENTITY_INCLUDE
#define MTL_MATRIX_IDENTITY_INCLUDE

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/matrix/diagonal_setup.hpp>

namespace mtl { namespace matrix {

namespace traits {

    // temporary solution, needs optimization
    template <typename Value= double>
    struct identity
    {
	typedef ::mtl::compressed2D<Value>  type;
    };
}

template <typename Value>
typename traits::identity<Value>::type
inline identity(std::size_t nrows, std::size_t ncols)
{
    typename traits::identity<Value>::type I(nrows, ncols);
    diagonal_setup(I, math::one(Value()));
    return I;
}


traits::identity<double>::type
inline identity(std::size_t nrows, std::size_t ncols)
{
    return identity<double>(nrows, ncols);
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_IDENTITY_INCLUDE
