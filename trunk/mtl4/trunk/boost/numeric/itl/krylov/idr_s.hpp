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

#ifndef ITL_IDR_S_INCLUDE
#define ITL_IDR_S_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>

namespace itl {


template < typename LinearOperator, typename Vector, 
	   typename LeftPreconditioner, typename RightPreconditioner, 
	   typename Iteration >
int idr_s(const LinearOperator &A, Vector &x, const Vector &b,
	  const LeftPreconditioner &L, const RightPreconditioner &R, 
	  Iteration& iter, size_t s)
{
    using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  Size;

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");

#if 0
    int                         n= size(x);
    const Scalar                zero= math::zero(b[0]), one= math::one(b[0]);
    Scalar                      omega;
    Vector                      x0(x), y(n), v(n), t(n);
    mtl::dense_vector<Vector>   dR(s, Vector(n, zero)), dX(s, Vector(n, zero)),
	                        P(s, Vector(n, zero));

    r= b - A * x;

    rand(P); // to be done !!!
    P[0]= r;
    orth(P);

    for (size_t k= 1; k < s; k++) {
	v= A * r;
	omega= dot(v, r) / dot(v, v);
	dX[k]= omega * r;
	dR[k]= -omega * v;
	x+= dX[k]; 
	r+= dR[k];
	if (iter.finished(r))
	    return iter;
	M[iall][k]= trans(P) * dR[k]; // TBD: column of matrix, trans(vec<vec>) * vec 
    }

    int oldest= 1;
    iter+= s;
    mtl::dense_vector<Scalar> m(trans(P) * r), c(s), dm(s); // TBD: trans(vec<vec>) * vec 

    while (! iter.finished(r)) {
       
	for (size_t k= 0; k < s+1; k++) {
	    c= solve(M, m);  // TBD: dense solver
	    q= -(dR * c);    // TBD: vec<vec> * vec
	    v= r + q;
	    if (k == 0) {
		t= A * v;
		omega= dot(t, v) / dot(t, t);
		dR[oldest]= q - omega * t;
		dX[oldest]= -(dX * v) + omega * v;
	    } else {
		dX[oldest]= -(dX * c) + omega * v;
		dR[oldest]= -(A * dX[oldest]);
	    }
	    r+= dR[oldest];
	    x+= dX[oldest];

	    if ((++iter).finished(r))
		return iter;

	    dm= trans(P) * dR[oldest];
	    M[iall][oldest]= dm;
	    m+= dm;
	    oldest= (oldest + 1) % s;
	}

    }
#endif
    return iter;
}


} // namespace itl

#endif // ITL_IDR_S_INCLUDE
