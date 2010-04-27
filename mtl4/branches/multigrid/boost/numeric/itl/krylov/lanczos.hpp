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

#ifndef ITL_LANCZOS_INCLUDE
#define ITL_LANCZOS_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/mtl/operation/dot.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>

namespace itl {

/// LANCZOS   without preconditioning
template < typename LinearOperator, typename Vector, 
	   typename Preconditioner, typename Iteration >
int lanczos(const LinearOperator &A, Vector &x, const Vector &b,
	const Preconditioner &M, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  size_type;
    Scalar     alpha(0), beta(0), delta(0), d(0), d_old(0), beta_0(0), zeta_old, zeta, delta_old, tol;
    Vector     v(resource(x)), c_old(resource(x)), q(resource(x)), c(resource(x)), q_last(resource(x)),
	       u(resource(x)), r(b - A * x), tmp(b-A*x);
    mtl::dense2D<Scalar>  T(A), V(A), W(A);
	   
    mtl::irange cols(0, num_cols(A)); // Intervals [0, n-1]
    V= 0;
    W= 0;
    T= 0;
    
    q=0; c=0; c_old=0;
    beta= beta_0= two_norm(r);
    q= r/beta;
    q_last=0;
    
    for (size_type i=0; i < num_cols(A); i++) {
	//step2
	u= A*q;
	alpha= dot(q,u);
	r= A*q-alpha*q-beta*q_last;
	beta= two_norm(r);
	q_last= q;
	q= r/beta;
	//step3
	d= alpha- delta*delta*d_old;
	if (iter.first())
	    zeta_old= beta/d;
	
	delta_old= delta;
	delta= beta/d;
// 	std::cout<< "d=" << d << "\n";
// 	std::cout<< "delta_old=" << delta_old << "\n";
// 	std::cout<< "delta=" << delta << "\n";
// 	std::cout<< "dzeta_old=" << zeta_old<< "\n";
	//step4
	zeta= -delta_old*zeta_old*delta/d;
	if ( !iter.first())
	    zeta_old= zeta;
	c= q- delta* c_old;
	c_old= c;
 	std::cout<< "zeta=" << zeta << "\t\t";  //in secend step to small ???
 	std::cout<< "c=" << c << "\n";
// 	std::cout<< "x=" << x << "\n";
	x+= zeta*c;
	std::cout<< "x=" << x << "\n";
	tol= abs(beta*zeta)/abs(beta_0);
// 	std::cout<< "i=" << i << "\t";
// 	std::cout<< "tol=" << tol << "\n";
	tmp=A*x;
// 	std::cout<< "A*x=" << tmp << "\n";
	tmp= b-tmp;
// 	std::cout<< "A*x-b=" << tmp << "\n";
	
        ++iter;
    }
//     std::cout<< "V=\n" << V << "\n";
//     std::cout<< "W=\n" << W << "\n";
//     std::cout<< "T=\n" << T << "\n";
//     std::cout<< "trans(W)*A*V=\n" << trans(W)*A*V << "\n";
    return iter;
}

} // namespace itl

#endif // ITL_LANCZOS_INCLUDE



