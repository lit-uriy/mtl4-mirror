// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_BICG_INCLUDE
#define ITL_BICG_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>

namespace itl {


template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicg(const LinearOperator &A, Vector &x, const Vector &b,
	 const Preconditioner &M, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    Scalar     rho_1, rho_2, alpha, beta;
    Vector     r(b - A * x), z(size(x)), p(size(x)), q(size(x)),
 	       r_tilde(r), z_tilde(size(x)), p_tilde(size(x)), q_tilde(size(x));

    while (! iter.finished(r)) {
	z= solve(M, r);
	z_tilde= adjoint_solve(M, r_tilde);
	rho_1= dot(z, r_tilde);

	if (rho_1 == 0.) {
	    iter.fail(2, "bicg breakdown");
	    break;
	}
	if (iter.first()) {
	    p= z;
	    p_tilde= z_tilde;
	} else {
	    beta= rho_1 / rho_2;      
	    p= z + beta * p;
	    p_tilde= z_tilde + beta * p_tilde;
	}
	q= A * p;
	q_tilde= adjoint(A) * p_tilde;
	alpha= rho_1 / dot(p_tilde, q);
	x+= alpha * p;
	r-= alpha * q;
	r_tilde-= alpha * q_tilde;

	rho_2= rho_1;
	++iter;
    }
    return iter.error_code();
}

} // namespace itl

#endif // ITL_BICG_INCLUDE









#if 0 // from IML++

	z= solve(M, r);
	ztilde= adjoint_solve(M, rtilde);
	rho_1= dot(z, rtilde);

	if (rho_1 ==

    if (rho_1(0) == 0) { 
      tol = norm(r) / normb;
      max_iter = i;
      return 2;
    }
    if (i == 1) {
      p = z;
      ptilde = ztilde;
    } else {
      beta(0) = rho_1(0) / rho_2(0);
      p = z + beta(0) * p;
      ptilde = ztilde + beta(0) * ptilde;
    }
    q = A * p;
    qtilde = A.trans_mult(ptilde);
    alpha(0) = rho_1(0) / dot(ptilde, q);
    x += alpha(0) * p;
    r -= alpha(0) * q;
    rtilde -= alpha(0) * qtilde;

    rho_2(0) = rho_1(0);
    if ((resid = norm(r) / normb) < tol) {
      tol = resid;
      max_iter = i;
      return 0;
    }
  }

  tol = resid;
  return 1;

#endif
