// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_CG_INCLUDE
#define ITL_CG_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>


namespace itl {

template < class LinearOperator, class HilbertSpaceX, class HilbertSpaceB, 
	   class Preconditioner, class Iteration >
int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
       const Preconditioner& M, Iteration& iter)
{
  typedef HilbertSpaceX TmpVec;
  typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;

  Scalar rho, rho_1, alpha, beta;
  TmpVec p(size(x)), q(size(x)), r(size(x)), z(size(x));
  
  r = b - A*x;

  while (! iter.finished(r)) {
      z = solve(M, r);
      rho = dot(r, z);
    
      if (iter.first())
	  p = z;
      else {
	  beta = rho / rho_1;
	  p = z + beta * p;
      }
      
      q = A * p;
      alpha = rho / dot(p, q);
      
      x += alpha * p;
      r -= alpha * q;
      rho_1 = rho;
      
      ++iter;
  }
  return iter.error_code();
}


} // namespace itl

#endif // ITL_CG_INCLUDE
