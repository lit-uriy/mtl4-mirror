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


#ifndef ITL_TFQMR_INCLUDE
#define ITL_TFQMR_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>


namespace itl {


template < typename Matrix, typename Vector, typename Iteration >
int tfqmr(const Matrix &A, Vector &x, const Vector &b,
                 Iteration& iter)
{
    using mtl::irange; using mtl::imax; using mtl::matrix::strict_upper;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  Size;

    if (size(b) == 0) throw mtl::logic_error("empty rhs vector");

    const Scalar                zero= math::zero(b[0]);
    Scalar                      theta(0), eta(0), tau, rho, rhon, sigma,
                        alpha, beta, c, m, tol;
    Size                        k, n(size(x));
    Vector                      r(n), u1(n), u2(n), y1(n), y2(n), w(n), d(n), v(n);

    d=0;
    k= 0;
    r= b - A*x;
        if (iter.finished(r))
                return iter.error_code();
    w= r;
    y1= r;
    v= A*y1;
    u1= v;
    tau= two_norm(r);
    rho= tau*tau;



// TFQMR iteration
    while(! iter.finished(tau)){
            sigma= dot(r,v);
        if (sigma == zero){
                return iter.fail(1, "tfgmr breakdown, sigma=0 #1");
            break;
        }
        alpha= rho/sigma;

        //inner loop
        for(int j=1;j<3;j++){
            if(j==2){
                y2= y1 - alpha*v;
                u2= A*y2;
            }
            m= 2*k-2+j;
            if(j==1){
                w= w - alpha*u1;
                d= y1+ (theta*theta*eta/alpha)*d;
              }
            else{
                w= w - alpha*u2;
                d= y2+ (theta*theta*eta/alpha)*d;
            }
            theta= two_norm(w)/tau;
            c= 1/(sqrt(1+theta*theta));
            tau= tau*theta*c;
            eta= c*c*alpha;
            x= x + eta*d;

            //try to terminate the iteration
            tol= tau*sqrt(m+1);
            //if(iter.finished(tol)){  //test is actually negligible
            //    return iter.error_code();
            //    break;
            //}
        }//end inner loop
        if (rho == zero){
            return iter.fail(1, "tfgmr breakdown, rho=0 #2");
            break;
        }
        rhon= dot(r,w);
        beta= rhon/rho;
        rho= rhon;
        y1= w + beta*y2;
        u1= A*y1;
        v= u1 + beta*(u2 + beta*v);

        ++iter;
         }

   return iter;
}
} // namespace itl

#endif // ITL_TFQMR_INCLUDE 
