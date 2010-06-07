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

#ifndef ITL_CG_INCLUDE
#define ITL_CG_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/itl/itl_fwd.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>

namespace itl {

/// Conjugate Gradients
template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB, 
	   typename Preconditioner, typename Iteration >
int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
       const Preconditioner& L, Iteration& iter)
{
  typedef HilbertSpaceX Vector;
  typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;

  Scalar rho(0), rho_1(0), alpha(0);
  Vector p(resource(x)), q(resource(x)), r(resource(x)), z(resource(x));
  
  r = b - A*x;
  while (! iter.finished(r)) {
      ++iter;
      z = solve(L, r);
      rho = dot(r, z);
    
      if (iter.first())
	  p = z;
      else 
	  p = z + (rho / rho_1) * p;
      
      q = A * p;
      alpha = rho / dot(p, q);
      
      x += alpha * p;
      r -= alpha * q;
      rho_1 = rho;
  }
  return iter;
}



/// Conjugate Gradients with ignored right preconditioner to unify interface
template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB, 
	   typename Preconditioner, typename RightPreconditioner, typename Iteration >
int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
       const Preconditioner& L, const RightPreconditioner&, Iteration& iter)
{
    return cg(A, x, b, L, iter);
}

} // namespace itl

#endif // ITL_CG_INCLUDE
