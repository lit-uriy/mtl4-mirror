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

#ifndef ITL_CG_INCLUDE
#define ITL_CG_INCLUDE

#include <cmath>
#include <cassert>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/itl/itl_fwd.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/mtl/operation/unroll.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>

namespace itl {

/// Conjugate Gradients without preconditioning
template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB, 
	   typename Iteration >
int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
       Iteration& iter)
{
    mtl::vampir_trace<6001> tracer;
    using std::abs; using mtl::conj;
    typedef HilbertSpaceX Vector;
    typedef typename mtl::Collection<HilbertSpaceX>::value_type Scalar;
    typedef typename Iteration::real                            Real;

    Scalar rho(0), rho_1(0), alpha(0);
    Vector p(resource(x)), q(resource(x)), r(resource(x)), z(resource(x));
  
    r = b - A*x;
    rho = dot(r, r);
    while (! iter.finished(Real(abs(rho)))) {
	++iter;
	if (iter.first())
	    p = r;
	else 
	    p = r + (rho / rho_1) * p;	   

	q = A * p;
	alpha = rho / dot(p, q);
      
	x += alpha * p;
	rho_1 = rho;
#if 1
	r -= alpha * q;
	rho = dot(r, r);
#endif

#if 0
	{
	    mtl::vampir_trace<9901> tracer;
	    //rho= Scalar(0);
	    Scalar rho0(0), rho1(0), rho2(0), rho3(0);
	    assert(size(r) % 4 == 0);
	    for (unsigned i= 0, i_max= size(r); i < i_max; i+=4) {
#if 0
		Scalar ri= r[i]-= alpha * q[i];
		rho+= conj(ri) * ri;
#else
		Scalar ri0= r[i]-= alpha * q[i];
		rho0+= conj(ri0) * ri0;
		Scalar ri1= r[i+1]-= alpha * q[i+1];
		rho1+= conj(ri1) * ri1;
		Scalar ri2= r[i+2]-= alpha * q[i+2];
		rho2+= conj(ri2) * ri2;
		Scalar ri3= r[i+3]-= alpha * q[i+3];
		rho3+= conj(ri3) * ri3;
#endif
	    }
	    rho= rho0 + rho1 + rho2 + rho3;
	}
#endif
	    
    }

    return iter;
}

/// Conjugate Gradients
template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB, 
	   typename Preconditioner, typename Iteration >
int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
       const Preconditioner& L, Iteration& iter)
{
    if (is_identity(L))
	return cg(A, x, b, iter);

    mtl::vampir_trace<6002> tracer;
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
