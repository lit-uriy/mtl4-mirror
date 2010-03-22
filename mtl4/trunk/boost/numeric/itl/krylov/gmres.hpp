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

    const Scalar                zero= math::zero(Scalar());
    Scalar                      rho, w1, w2, nu, hr;
    Size                        k, n(size(x)), kmax(std::min(size(x), kmax_in));
    Vector                      r0(b - A *x), r(solve(L,r0)), va(resource(x)), va0(resource(x)), va00(resource(x));
    mtl::dense_vector<Scalar>   s(kmax+1, zero), c(kmax+1, zero), g(kmax+1, zero), y(kmax);  // replicated in distributed solvers 
    mtl::multi_vector<Vector>   V(Vector(resource(x), zero), kmax+1); 
    mtl::dense2D<Scalar>        H(kmax+1, kmax);                                             // replicated in distributed solvers 
    irange                      range_n(0, n);

    rho= g[0]= two_norm(r);
    if (iter.finished(rho))
	return iter;
    V.vector(0)= r / rho;

    // GMRES iteration
    for (k= 0; rho >= iter.atol() && k < kmax; k++, ++iter) {
	// std::cout << "GMRES full: k == " << k << ", rho == " << rho << ", x == " << x << '\n';
        va0= A * solve(R, V.vector(k));
        V.vector(k+1)= va= solve(L,va0);
	// orth(V, V[k+1], false); 
        // modified Gram Schmidt method
        for (Size j= 0; j < k+1; j++) {
	    H[j][k]= dot(V.vector(j), V.vector(k+1));
	    V.vector(k+1)-= H[j][k] * V.vector(j);
        }

        H[k+1][k]= two_norm(V.vector(k+1));
        //reorthogonalize
        for(Size j= 0; j < k+1; j++) {
	    hr= dot(V.vector(k+1), V.vector(j));
            H[j][k]+= hr;
            V.vector(k+1)-= hr * V.vector(j);
        }
        H[k+1][k]= two_norm(V.vector(k+1));
	if (H[k+1][k] != zero)                // watch for breakdown    
            V.vector(k+1)*= 1. / H[k+1][k];

        //k givensrotationen
	for(Size i= 0; i < k; i++) {
	    w1= c[i]*H[i][k]-s[i]*H[i+1][k];  // shouldn't c and s depend on H?
	    w2= s[i]*H[i][k]+c[i]*H[i+1][k];
	    H[i][k]= w1;
	    H[i+1][k]= w2;
	}
        nu= sqrt(H[k][k]*H[k][k]+H[k+1][k]*H[k+1][k]);
        if(nu != zero){
            c[k]=  H[k][k]/nu;
            s[k]= -H[k+1][k]/nu;
            H[k][k]=c[k]*H[k][k]-s[k]*H[k+1][k];
            H[k+1][k]=0;
	    w1= c[k]*g[k]-s[k]*g[k+1]; //given's rotation on solution
            w2= s[k]*g[k]+c[k]*g[k+1];
            g[k]= w1;
            g[k+1]= w2;
        }
	rho= abs(g[k+1]);
    }

    // iteration is finished -> compute x: solve H*y=g as far as rank of H allows
    irange                  range(k);
    for (bool solved= false; !solved && !range.empty(); --range) {
	try {
	    y[range]= lu_solve(H[range][range], g[range]); 
	} catch (mtl::matrix_singular) { continue; } // if singular then try with sub-matrix
	solved= true;
    }
    assert(!range.empty()); // Can H[0][0] be zero???
    if (range.finish() < k)
	std::cerr << "GMRES orhogonalized with " << k << " vectors but matrix singular, can only use " << range.finish() << " vectors!\n";

    x+= solve(R, Vector(V.vector(range)*y[range]));

    r= b - A*x;
    if (!iter.finished(r))
        return iter.fail(2, "GMRES does not converge");
    return iter;
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

