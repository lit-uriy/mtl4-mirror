// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_BICGSTAB_INCLUDE
#define ITL_BICGSTAB_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>

namespace itl {

template < class LinearOperator, class HilbertSpaceX, class HilbertSpaceB, 
	   class Preconditioner, class Iteration >
int bicgstab(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
	     const Preconditioner& M, Iteration& iter)
{
  typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;
  typedef HilbertSpaceX                                       Vector;

  Scalar     rho_1(0), rho_2(0), alpha(0), beta(0), omega(0);
  Vector     p(size(x)), phat(size(x)), s(size(x)), shat(size(x)), 
             t(size(x)), v(size(x)), r(size(x)), rtilde(size(x));

  r = b - A * x;
  rtilde = r;

  while (! iter.finished(r)) {
    
    rho_1 = dot(rtilde, r); // switched back
    if (rho_1 == Scalar(0.)) {
      iter.fail(2, "bicg breakdown #1");
      break;
    }
    
    if (iter.first())
      p = r;
    else {
      if (omega == Scalar(0.)) {
	iter.fail(3, "bicg breakdown #2");
	break;
      }
      beta = (rho_1 / rho_2) * (alpha / omega);
      p = r + beta * (p - omega * v);
    }
    phat = solve(M, p);
    v = A * phat;

    alpha = rho_1 / dot(rtilde, v); // switched 
    s = r - alpha * v;
    
    if (iter.finished(s)) {
      x += alpha * phat;
      break;
    }
    shat = solve(M, s);
    t = A * shat;
    omega = dot(t, s) / dot(t, t); // switched back
    
    x += omega * shat + alpha * phat;
    r = s - omega * t;
    
    rho_2 = rho_1;
    
    ++iter;
  }

  return iter.error_code();
}


} // namespace itl

#endif // ITL_BICGSTAB_INCLUDE
