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
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/dot.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>

namespace itl {

/// LANCZOS   without preconditioning
template < typename LinearOperator, typename Vector, typename Iteration >
int lanczos(const LinearOperator &A, Vector &x, Iteration& iter)
{
    using std::min; using mtl::irange; using mtl::imax;
    typedef typename mtl::Collection<Vector>::value_type Scalar;
    typedef typename mtl::Collection<Vector>::size_type  size_type;
    Scalar     alpha(0), beta(0), delta(0), d(0), d_old(0), beta_0(0), zeta_old, zeta, delta_old, tol;
    Vector     a(resource(x)), b(size(x)-1), q_c(resource(x)),q_old(resource(x)), tmp(resource(x));  //a maindiagonal, b maindiagonal+-1
    mtl::dense2D<Scalar>  Q(size(x), size(x));
    
	   
   //TODO If A is tridiagonal
   
   const Scalar IM= sqrt(-1), zero= math::zero(x[0]);  // ????
   size_type n= std::min(size(x), iter.max()), n_vec(0);//#vec for orthogonalization;
   a= zero; b= zero;
   Vector w_old(n, zero), w_c(n, zero);
   irange r(0, imax);
   Q= zero;
//    std::cout<< "Q=\n" << Q << "\n";
   
//    std::cout<< "n=" << n << "\n";
    
   w_old[0]= 1;  // evtl one must tested
   bool second(false);
   
   //starting vector
   q_c= x/two_norm(x);
   
   Q[r][0]= q_c;
   
   for (size_type j= 0; j < n; j++) {
        tmp= A*conj(q_c);
	a[j]= dot(q_c,tmp);
	if (j == 0)
	    x= tmp - a[j] * q_c;
	else
	    x= tmp - a[j] * q_c - b[j-1] * q_old;
	
	if (j < n-1) {
	    std::cout<< "j=" << j << "\n";
	    b[j]= two_norm(x);
	    if (j > 1) {  //compute orthogonality estimates
		

		
	    } //endif(j>1)
	} //endif (j<n-1)
	
	
	
        ++iter;
    } //endfor (j=0;...)

    return iter;
}

} // namespace itl

#endif // ITL_LANCZOS_INCLUDE



