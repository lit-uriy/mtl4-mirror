// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_MULTIPLIER_INCLUDE
#define MTL_MATRIX_MULTIPLIER_INCLUDE

#include <boost/numeric/mtl/vector/assigner.hpp>
#include <boost/numeric/mtl/vector/decrementer.hpp>
#include <boost/numeric/mtl/vector/incrementer.hpp>
#include <boost/numeric/mtl/operation/assign_mode.hpp>

namespace mtl { namespace matrix {

template <typename Matrix, typename VectorIn>
struct multiplier
  : mtl::vector::assigner<multiplier<Matrix, VectorIn> >,
    mtl::vector::incrementer<multiplier<Matrix, VectorIn> >,
    mtl::vector::decrementer<multiplier<Matrix, VectorIn> >
{
    multiplier(const Matrix& A, const VectorIn& v) : A(A), v(v) {}

    template <typename VectorOut>
    void assign_to(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::assign_sum());
    }

    template <typename VectorOut>
    void increment_it(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::plus_sum());
    }

    template <typename VectorOut>
    void decrement_it(VectorOut& w) const
    {
	A.mult(v, w, mtl::assign::minus_sum());
    }

    const Matrix&   A;
    const VectorIn& v;
};


}} // namespace mtl::matrix

#endif // MTL_MATRIX_MULTIPLIER_INCLUDE
