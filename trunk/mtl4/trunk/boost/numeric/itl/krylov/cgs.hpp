// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_CGS_INCLUDE
#define ITL_CGS_INCLUDE

namespace itl {


template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int cgs(const LinearOperator &A, Vector &x, const Vector &b,
	const Preconditioner &M, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    Scalar     rho_1, rho_2, alpha, beta;
    Vector     p(size(x)), phat(size(x)), q(size(x)), qhat(size(x)), vhat(size(x)),
	       u(size(x)), uhat(size(x)), r(b - A * x), rtilde= r;


    while (! iter.finished(r)) {
	rho_1= dot(rtilde, r);

	if (rho_1 == 0.) {
	    iter.fail(2, "cgs breakdown");
	    break;
	}

	if (iter.first())
	    p= u= r;
	else {
	    beta = rho_1 / rho_2;
	    u= r + beta * q;
	    p= u + beta * (q + beta * p);
	}

        vhat= A * solve(M, p);
	alpha = rho_1 / dot(rtilde, vhat);
	q= u - alpha * vhat;

	u+= q;
	uhat= solve(M, u);
	
	x+= alpha * uhat;
	qhat= A * uhat;
	r-= alpha * qhat;

	rho_2 = rho_1;
	++iter;
    }
    return iter.error_code();
}

} // namespace itl

#endif // ITL_CGS_INCLUDE










#if 0

 Real resid;
  Vector rho_1(1), rho_2(1), alpha(1), beta(1);
  Vector p, phat, q, qhat, vhat, u, uhat;

  Real normb = norm(b);
  Vector r = b - A*x;
  Vector rtilde = r;

  if (normb == 0.0)
    normb = 1;
  
  if ((resid = norm(r) / normb) <= tol) {
    tol = resid;
    max_iter = 0;
    return 0;
  }

  for (int i = 1; i <= max_iter; i++) {
    rho_1(0) = dot(rtilde, r);
    if (rho_1(0) == 0) {
      tol = norm(r) / normb;
      return 2;
    }
    if (i == 1) {
      u = r;
      p = u;
    } else {
      beta(0) = rho_1(0) / rho_2(0);
      u = r + beta(0) * q;
      p = u + beta(0) * (q + beta(0) * p);
    }
    phat = M.solve(p);
    vhat = A*phat;
    alpha(0) = rho_1(0) / dot(rtilde, vhat);
    q = u - alpha(0) * vhat;
    uhat = M.solve(u + q);
    x += alpha(0) * uhat;
    qhat = A * uhat;
    r -= alpha(0) * qhat;
    rho_2(0) = rho_1(0);
    if ((resid = norm(r) / normb) < tol) {
      tol = resid;
      max_iter = i;
      return 0;


#endif
