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

#ifndef MTL_MATRIX_REORDER_INCLUDE
#define MTL_MATRIX_REORDER_INCLUDE

#include <algorithm>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/reorder_ref.hpp>

namespace mtl { namespace matrix {

    
namespace traits {

    /// Return type of mtl::matrix::reorder	
    template <typename Value= short>
    struct reorder
    {
	typedef ::mtl::compressed2D<Value>  type;
    };
}


template <typename Value, typename ReorderVector>
typename traits::reorder<Value>::type
reorder(const ReorderVector& v, std::size_t cols= 0)
{
    typename traits::reorder<Value>::type A;
    reorder_ref(v, A, cols);
    return A;
}


/// Computes reordering matrix from corresponding vector
template <typename ReorderVector>
typename traits::reorder<>::type
inline reorder(const ReorderVector& v, std::size_t cols= 0)
{
    return reorder<short>(v, cols);
}


}} // namespace mtl::matrix

namespace mtl { namespace vector {

    /// Import into vector namespace; see \ref mtl::matrix::reorder
    using mtl::matrix::reorder;

}} // namespace mtl::vector

#endif // MTL_MATRIX_REORDER_INCLUDE
