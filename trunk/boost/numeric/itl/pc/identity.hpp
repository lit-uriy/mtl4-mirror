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

#ifndef ITL_PC_IDENTITY_INCLUDE
#define ITL_PC_IDENTITY_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>

namespace itl { namespace pc {

template <typename Matrix>
class identity
{
  public:
    typedef typename mtl::Collection<Matrix>::value_type  value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef identity                                      self;

    identity(const Matrix&) {}

    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	mtl::vampir_trace<5032> tracer;
	return x;
    }

    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	mtl::vampir_trace<5034> tracer;
	return x;
    }
}; 


template <typename Matrix, typename Vector>
inline Vector solve(const identity<Matrix>&, const Vector& x)
{
    mtl::vampir_trace<5031> tracer;
    return x;
}

template <typename Matrix, typename Vector>
inline Vector adjoint_solve(const identity<Matrix>&, const Vector& x)
{
    mtl::vampir_trace<5033> tracer;
    return x;
}



}} // namespace itl::pc

#endif // ITL_PC_IDENTITY_INCLUDE
