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

#ifndef ITL_BICGSTAB_2_INCLUDE
#define ITL_BICGSTAB_2_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/itl/pc/identity.hpp>

namespace itl {

/// Bi-Conjugate Gradient Stabilized(2)
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicgstab_2(const LinearOperator &A, Vector &x, const Vector &b,
	       const Preconditioner &L, Iteration& iter)
{
    using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  Size;

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");

    const size_t                l= 2;
    const Scalar                zero= math::zero(Scalar()), one= math::one(Scalar());
    Vector                      x0(resource(x)), y(resource(x));
    mtl::vector::dense_vector<Vector>   r_hat(l+1,Vector(resource(x))), u_hat(l+1,Vector(resource(x)));

    // shift problem 
    x0= zero;
    r_hat[0]= b;
    if (two_norm(x) != zero) {
	r_hat[0]-= A * x;
	x0= x;
	x= zero;
    }

    Vector  r0_tilde(r_hat[0]/two_norm(r_hat[0]));
    y= solve(L, r_hat[0]);
    r_hat[0]= y;
    u_hat[0]= zero;

    Scalar                      rho_0(one), rho_1(zero), alpha(zero), Gamma(zero), beta(zero), omega(one); 
    mtl::matrix::dense2D<Scalar>        tau(l+1, l+1);
    mtl::vector::dense_vector<Scalar>   sigma(l+1), gamma(l+1), gamma_a(l+1), gamma_aa(l+1);

    while (! iter.finished(r_hat[0])) {
	++iter;
	rho_0= -omega * rho_0;

	for (Size j= 0; j < l; ++j) {
	    rho_1= dot(r0_tilde, r_hat[j]); 
	    beta= alpha * rho_1/rho_0; rho_0= rho_1;

	    for (Size i= 0; i <= j; ++i)
		u_hat[i]= r_hat[i] - beta * u_hat[i];
      
	    y= A * u_hat[j];
	    u_hat[j+1]= solve(L, y);
	    Gamma= dot(r0_tilde, u_hat[j+1]); 
	    alpha= rho_0 / Gamma;

	    for (Size i= 0; i <= j; ++i)
		r_hat[i]-= alpha * u_hat[i+1];
      
	    if (iter.finished(r_hat[j])) {
		x+= x0;
		return iter;
	    }

	    y= A * r_hat[j]; 
	    r_hat[j+1]= solve(L, y);
	    x+= alpha * u_hat[0];
	}

	// mod GS (MR part)
	mtl::vector::dense_vector<Vector>   r_hat_tail(r_hat[irange(1, imax)]);
	tau[irange(1, imax)][irange(1, imax)]= orthogonalize_factors(r_hat_tail);
	for (Size j= 1; j <= l; ++j) 
	    gamma_a[j]= dot(r_hat[j], r_hat[0]) / tau[j][j];

	gamma[l]= gamma_a[l]; omega= gamma[l];
	if (omega == zero) return iter.fail(3, "bicg breakdown #2");
		

	gamma[1]= gamma_a[1] - tau[1][2] * gamma[2];
	gamma_aa[irange(1, l)]= strict_upper(tau[irange(1, l)][irange(1, l)]) * gamma[irange(2, l+1)] + gamma[irange(2, l+1)];

	x+= gamma[1] * r_hat[0];
	r_hat[0]-= gamma_a[l] * r_hat[l];
	u_hat[0]-= gamma[l] * u_hat[l];
	for (Size j=1; j < l; ++j) {
	    u_hat[0] -= gamma[j] * u_hat[j];
	    x+= gamma_aa[j] * r_hat[j];
	    r_hat[0] -= gamma_a[j] * r_hat[j];
	}
    }
    x+= x0; // convert to real solution and undo shift
    return iter;
}




#if 0
/// Bi-Conjugate Gradient Stabilized(2)
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicgstab_2(const LinearOperator &A, Vector &x, const Vector &b,
	       const Preconditioner &, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    const Scalar zero= math::zero(Scalar()), one= math::one(Scalar());
    Scalar     alpha(zero), beta, gamma, mu, nu, rho_0(one), rho_1, tau, omega_1, omega_2(one);
    Vector     r(b - A * x), r_0(r), r_i(r), x_i(x), 
	       s(resource(x)), t(resource(x)), u(resource(x), zero), v(resource(x)), w(resource(x));

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");
    while ( ! iter.finished(r)) {
	++iter;
	rho_0*= -omega_2;
	// z= solve(M, r); z_tilde= solve(M, r_tilde); ???

	rho_1= dot(r_0, r_i);       // or rho_1= dot(z, r_tilde) ???
	beta= alpha * rho_1 / rho_0; rho_0= rho_1;
	u= r_i - beta * u;
	v= A * u;
	gamma= dot(v, r_0); alpha= rho_0 / gamma;
	r= r_i - alpha * v;
	s= A * r;
	x= x_i + alpha * u;

	rho_1= dot(r_0, s); beta= alpha * rho_1 / rho_0; rho_0= rho_1;
	v= s - beta * v;
	w= A * v;
	gamma= dot(w, r_0); alpha= rho_0 / gamma;
	u= r - beta * u;
	r-= alpha * v;
	s-= alpha * w;
	t= A * s;

	omega_1= dot(r, s); mu= dot(s, s); nu= dot(s, t); tau= dot(t, t);
	omega_2= dot(r, t); tau-= nu * nu / mu; omega_2= (omega_2 - nu * omega_1 / mu) / tau;
	omega_1= (omega_1 - nu * omega_2) / mu;
	x_i= x + omega_1 * r + omega_2 * s + alpha * u;
	r_i= r - omega_1 * s - omega_2 * t;
	u-= omega_1 * v + omega_2 * w;
    }
    return iter;
}
#endif

} // namespace itl

#endif // ITL_BICGSTAB_2_INCLUDE






