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

// Written by Cornelius Steinhardt


#ifndef ITL_GMRES_INCLUDE
#define ITL_GMRES_INCLUDE

#include <algorithm>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/multi_vector.hpp>
#include <boost/numeric/mtl/matrix/strict_upper.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>

namespace itl {

/// Generalized Minimal Residual method (without restart)
/** It computes at most kmax_in iterations (or size(x) depending on what is smaller) 
    regardless on whether the termination criterion is reached or not.   **/
template < typename Matrix, typename Vector, typename LeftPreconditioner, typename RightPreconditioner, typename Iteration >
int gmres_full(const Matrix &A, Vector &x, const Vector &b,
               LeftPreconditioner &L, RightPreconditioner &R,
               Iteration& iter, typename mtl::Collection<Vector>::size_type kmax_in)
{
    using mtl::irange; using mtl::iall; using mtl::matrix::strict_upper; using std::abs; using std::sqrt;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  Size;

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");

    const Scalar                zero= math::zero(b[0]);
    Scalar                      rho, w1, w2, nu, hr;
    Size                        k, n(size(x)), kmax(std::min(size(x), kmax_in));
    Vector                      r0(b - A *x), r(solve(L,r0)), s(kmax+1),
                                c(kmax+1), g(kmax+1), va(resource(x)), va0(resource(x)), va00(resource(x));
    mtl::multi_vector<Vector>   v(Vector(resource(x), zero), kmax+1); 
    mtl::dense2D<Scalar>        h(kmax+1, kmax);
    irange                      range_n(0, n);

    rho= g[0]= two_norm(r);
    if (iter.finished(rho))
	return iter;
    v.vector(0)= r / rho;

    // GMRES iteration
    for (k= 0; rho >= iter.atol() && k < kmax; k++, ++iter) {
	// std::cout << "GMRES full: k == " << k << ", rho == " << rho << ", x == " << x << '\n';
        va0= A * solve(R, v.vector(k));
        v.vector(k+1)= va= solve(L,va0);
	// orth(v, v[k+1], false); 
        // modified Gram Schmidt method
        for (Size j= 0; j < k+1; j++) {
	    h[j][k]= dot(v.vector(j), v.vector(k+1));
	    v.vector(k+1)-= h[j][k] * v.vector(j);
        }

        h[k+1][k]= two_norm(v.vector(k+1));
        //reorthogonalize
        for(Size j= 0; j < k+1; j++) {
	    hr= dot(v.vector(k+1), v.vector(j));
            h[j][k]+= hr;
            v.vector(k+1)-= hr * v.vector(j);
        }
        h[k+1][k]= two_norm(v.vector(k+1));
        //watch for breakdown
        if (h[k+1][k] != zero)
            v.vector(k+1)*= 1. / h[k+1][k];

        //k givensrotationen
	for(Size i= 0; i < k; i++) {
	    w1= c[i]*h[i][k]-s[i]*h[i+1][k];
	    w2= s[i]*h[i][k]+c[i]*h[i+1][k];
	    h[i][k]= w1;
	    h[i+1][k]= w2;
	}
        nu= sqrt(h[k][k]*h[k][k]+h[k+1][k]*h[k+1][k]);
        if(nu != zero){
            c[k]=  h[k][k]/nu;
            s[k]= -h[k+1][k]/nu;
            h[k][k]=c[k]*h[k][k]-s[k]*h[k+1][k];
            h[k+1][k]=0;
	    w1= c[k]*g[k]-s[k]*g[k+1];//givensrotation on solutionparameters
            w2= s[k]*g[k]+c[k]*g[k+1];//givensrotation on solutionparameters
            g[k]= w1;
            g[k+1]= w2;
        }
	rho= abs(g[k+1]);
    }

    // iteration is finished -> compute x: solve H*y=g
    irange                  range_k(0, k);
    // mtl::dense2D<Scalar>    h_a(h[range_k][range_k]), v_a(v[iall][range_k]);
    mtl::dense2D<Scalar>    v_a(n, k);

    for(Size j = 0 ; j < k; j++) {
	for(Size i = 0; i < n; i++)
            v_a[i][j]= v[i][j];
    }

    Vector                  g_a(g[range_k]), y;
    try {
	y= lu_solve(h[range_k][range_k], g_a);
    } catch (mtl::matrix_singular e) {
	return iter.fail(2, "GMRES sub-system singular");
    }

    // x+= solve(R, Vector(v.vector(range_k)*y));
    x+= solve(R, Vector(v_a*y));

    r= b - A*x;
    if (!iter.finished(r))
        return iter.fail(2, "GMRES does not converge");
    return iter.error_code();
}

/// Generalized Minimal Residual method with restart
template < typename Matrix, typename Vector, typename LeftPreconditioner,
           typename RightPreconditioner, typename Iteration >
int gmres(const Matrix &A, Vector &x, const Vector &b,
          LeftPreconditioner &L, RightPreconditioner &R,
	  Iteration& iter, typename mtl::Collection<Vector>::size_type restart)
{
    iter.set_quite(true);
    while (!iter.finished()) 
	gmres_full(A, x, b, L, R, iter, restart);
    iter.set_quite(false);
    return iter;
}



} // namespace itl

#endif // ITL_GMRES_INCLUDE

