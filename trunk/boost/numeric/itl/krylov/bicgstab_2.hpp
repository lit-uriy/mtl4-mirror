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
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/matrix/strict_upper.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/operation/orth.hpp>
#include <boost/numeric/mtl/operation/lazy.hpp>

namespace itl {

#if 1
/// Bi-Conjugate Gradient Stabilized(2)
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicgstab_2(const LinearOperator &A, Vector &x, const Vector &b,
	       const Preconditioner &L, Iteration& iter)
{
    using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper; using mtl::lazy;
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

    Scalar                      rho_0(one), rho_1(zero), alpha(zero), Gamma(zero), beta(zero), omega(one), gamma_aa(zero); 
    mtl::matrix::dense2D<Scalar>        tau(l+1, l+1);
    mtl::vector::dense_vector<Scalar>   sigma(l+1), gamma(l+1), gamma_a(l+1);

    while (! iter.finished(r_hat[0])) {
	++iter;
	rho_0= -omega * rho_0;

	for (Size j= 0; j < 2; ++j) {
	    rho_1= dot(r0_tilde, r_hat[j]); 
	    beta= alpha * rho_1/rho_0; rho_0= rho_1;

	    for (Size i= 0; i <= j; ++i)
		u_hat[i]= r_hat[i] - beta * u_hat[i];
      
	    y= A * u_hat[j];
	    u_hat[j+1]= solve(L, y);
	    Gamma= dot(r0_tilde, u_hat[j+1]); 
	    // (lazy(u_hat[j+1])= solve(L, y)) || (lazy(Gamma)= lazy_dot(r0_tilde, u_hat[j+1])); // not faster
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
	irange  i1m(1, imax);
	mtl::vector::dense_vector<Vector>   r_hat_tail(r_hat[i1m]);
	tau[i1m][i1m]= orthogonalize_factors(r_hat_tail);
	for (Size j= 1; j <= l; ++j) 
	    gamma_a[j]= dot(r_hat[j], r_hat[0]) / tau[j][j];

	gamma[l]= gamma_a[l]; omega= gamma[l];
	if (omega == zero) return iter.fail(3, "bicg breakdown #2");
		
	gamma[1]= gamma_a[1] - tau[1][2] * gamma[2];
	gamma_aa= (tau[1][1] + one) * gamma[2];

	x+= gamma[1] * r_hat[0];
	r_hat[0]-= gamma_a[2] * r_hat[2];
	u_hat[0]-= gamma[2] * u_hat[2];
	u_hat[0]-= gamma[1] * u_hat[1];
	x+= gamma_aa * r_hat[1];
	r_hat[0] -= gamma_a[1] * r_hat[1];
    }
    x+= x0; // convert to real solution and undo shift
    return iter;
}

#else

/// Bi-Conjugate Gradient Stabilized(2)
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicgstab_2(const LinearOperator& A, Vector& x, const Vector& b,
	       const Preconditioner& L, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    const Scalar zero= math::zero(Scalar()), one= math::one(Scalar());
    Scalar     alpha(zero), beta, gamma, mu, nu, rho_0(one), rho_1, tau, omega_1, omega_2(one);
    Vector     r(b - A * x), r_0(r), r_i(r), x_i(x), p(resource(x)), v_tilde(resource(x)),
	       s(resource(x)), t(resource(x)), u(resource(x), zero), v(resource(x)), w(resource(x));

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");
    while ( ! iter.finished(r)) {
	++iter;
	rho_0*= -omega_2;
	// z= solve(M, r); z_tilde= solve(M, r_tilde); ???

	r_i= solve(L, r_0); // pg

	rho_1= dot(r_0, r_i);       // or rho_1= dot(z, r_tilde) ???
	beta= alpha * rho_1 / rho_0; rho_0= rho_1;
	u= r_i - beta * u;
	v= A * u;
	
	v_tilde= solve(L, v);


	gamma= dot(v_tilde, r_0); alpha= rho_0 / gamma;
	r= r_i - alpha * v_tilde;
	

	s= A * r;
	s_tilde= solve(L, s); // s === y

	x= x_i + alpha * u;

	rho_1= dot(r_0, s_tilde); beta= alpha * rho_1 / rho_0; rho_0= rho_1;
	v= s - beta * v;
	w= A * v;  // w === y
	w_tilde= solve(L, w);
	
	gamma= dot(w_tilde, r_0); alpha= rho_0 / gamma;
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


//Vorschlag von Cornelius
#if 0

/// Bi-Conjugate Gradient Stabilized(2)
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int bicgstab_2(const LinearOperator &A, Vector &x, const Vector &b,
	       const Preconditioner &M, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    const Scalar zero= math::zero(Scalar()), one= math::one(Scalar());
    Scalar     p_0(one),
	       p_1,
	       alpha(one),
	       beta,
	       sigma,
	       omega(one),
	       a,e,c,d,det, y_1, y_2, z_1, z_2
	       ;
    Vector     r(b - A * x), r_0(solve(M,r)), r_tilde(r_0),  
	       u_0(resource(x), zero),
	       u_1(resource(x), zero),
	       u_2(resource(x), zero),
	       r_1(resource(x), zero),
	       r_2(resource(x), zero)
	       ;
    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");
    
    while ( ! iter.finished(r_0)) {
      ++iter;
      
 //     std::cout<< "p_0=" << p_0 << "  norm_r_0="<< two_norm(r_0)<< "\n";
      p_0*= -omega;
      p_1= dot(r_0, r_tilde);
      beta= alpha * p_1 / p_0;
      p_0= p_1;
      u_0= r_0 - beta * u_0;
      u_1= solve(M, Vector( A * u_0 ));
      sigma= dot(u_1, r_tilde);
      alpha= p_1 / sigma;
      x= x + alpha * u_0;
      r_0= r_0 - alpha * u_1;
      r_1= solve(M, Vector( A * r_0 ));
      p_1= dot(r_1, r_tilde);
      beta= alpha * p_1 / p_0;
      p_0= p_1;
      u_0= r_0 - beta * u_0;
      u_1= r_1 - beta * u_1;
      u_2= solve(M, Vector( A * u_1 ));
      sigma= dot(u_2, r_tilde);
      alpha= p_1 / sigma;
      x= x + alpha * u_0;
      r_0= r_0 - alpha * u_1;
      r_1= r_1 - alpha * u_2;
      r_2= solve(M, Vector( A * r_1 ));
      
      a= dot(r_1, r_1);
      e= dot(r_2, r_1);
      c= e;
      d= dot(r_2, r_2);
      y_1= dot(r_0, r_1);
      y_2= dot(r_0, r_2);
      det= 1 / ( a * d - e * c );
      
      z_1= det * (d * y_1 - e * y_2);
      z_2= det * (a * y_2 - c * y_1);
      
      
      omega= z_2;
      
      u_0= u_0 - z_1 * u_1;
      
      x= x + z_1 * r_0;
      r_0= r_0 - z_1 * r_1;
      u_0= u_0 - z_2 * u_2;
      x= x + z_2 * r_1;
      r_0= r_0 - z_2 * r_2;
        
    }
    return iter;
}


#endif





} // namespace itl

#endif // ITL_BICGSTAB_2_INCLUDE






