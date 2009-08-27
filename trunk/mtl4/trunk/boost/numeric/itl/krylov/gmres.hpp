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

namespace itl {


template < typename Matrix, typename Vector, typename Iteration >
int gmres(const Matrix &A, Vector &x, const Vector &b,
	  Iteration& iter, typename mtl::Collection<Vector>::size_type kmax_in)
{
    using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper; using std::abs;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  Size;

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");

    const Scalar                zero= math::zero(b[0]);
    Scalar                      rho, hr, w1, w2, nu;
    Size                        k, n(size(x)), kmax(std::min(size(x), kmax_in));

    Vector                      r(n),r0(n),s(kmax+1), c(kmax+1), g(kmax+1), vi(n), va(n);
    mtl::multi_vector<Vector>   v(kmax+1, Vector(n, zero)), h(kmax, Vector(kmax+1, zero));
    irange                      range_n(0, n);

    r0= b - A * x;
    if (iter.finished(r0))
	return iter;

    rho= g[0]= two_norm(r0);
    v.vector(0)= r0 / rho;

    // GMRES iteration
    for (k= 0; !iter.finished(rho) && k < kmax; k++, ++iter) {

	// vi= v.vector(k)[range_n];
        for (Size i= 0; i < n; i++)
	    vi[i]= v[i][k];

        //v.vector(k+1)[irange(0, n)]= va= A * vi;
	va= A * vi;
        for (Size i= 0; i < n; i++)
	    v[i][k+1]=va[i];

        // modified Gram Schmidt method
        for (Size j= 0; j < k+1; j++) {
            for (Size i= 0; i < n; i++)
                vi[i]= v[i][j];
	    // h[j][k]= dot(v.vector(j)[range_n], va);
            h[j][k]= dot(vi,va);
	    // v.vector(k+1)[range_n]-= ele_prod(h.vector(k), v.vector(j)[range_n]);
            for (Size i= 0;i < n; i++)
                v[i][k+1]-= h[j][k]*vi[i];
        }

	// h[k+1][k]= two_norm(v.vector(k+1)[range_n]);
        for(Size i= 0; i < n; i++)
            vi[i]=v[i][k+1];
        h[k+1][k]= two_norm(vi);

        //reorthogonalize
        for(Size j=0; j < k+1;j++){
            for(Size i= 0; i < n;i++)
                va[i]= v[i][j];
            hr= dot(vi,va);
            h[j][k]= h[j][k] + hr;
            vi= vi - hr*va;
        }
        h[k+1][k]= two_norm(vi);

        //watch for breakdown
        if(h[k+1][k] != zero)
            vi = vi/h[k+1][k];
        for(Size i= 0; i < n;i++)
            v[i][k+1]=vi[i];

        //k givensrotationen
        if(k>0)
            for(Size i=0; i < k;i++) {
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

	rho= abs(g[k]);
    }

    //iterator is finish -> compute x
    // solve H*y=g
    Size        max_iter= k;
    irange      range_iter(0, max_iter);
    // dense2D     h_a(h[range_iter][range_iter]), v_a(v[iall][range_iter]);
    // Vector      g_a(g[range_iter]), y(lu_solve_new(h_a, g_a))


    mtl::dense2D<Scalar>   h_a(max_iter, max_iter), v_a(n, max_iter);
    Vector                 g_a(max_iter), y(max_iter);
    for (Size i = 0; i < max_iter;i++){
        g_a[i]=g[i];
        for(Size j = 0 ; j< max_iter ;j++)
            h_a[i][j]= h[i][j];
    }
    y=lu_solve_new(h_a, g_a);

    for(Size i = 0; i < n ;i++){
        for(Size j = 0 ; j< max_iter ;j++)
            v_a[i][j]=v[i][j];
    }
    //calculation of solution
    // x+= v[iall][range_iter] * lu_solve_new(h[range_iter][range_iter], g[range_iter]);
    x+= v_a * y;
    r= b - A*x;
    if(two_norm(r0)<two_norm(r))
        return iter.fail(2, "GMRES does not converge");

    return iter;
}

} // namespace itl

#endif // ITL_GMRES_INCLUDE
