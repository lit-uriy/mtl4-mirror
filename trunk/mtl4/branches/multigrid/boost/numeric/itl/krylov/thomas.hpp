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

#ifndef ITL_THOMAS_INCLUDE
#define ITL_THOMAS_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>

namespace itl {

/// THOMAS  algorithm for tridiagonal matrix without preconditioning
template < typename LinearOperator, typename Vector, 
	  typename Iteration >
int thomas(const LinearOperator &A, Vector &x, const Vector &b, Iteration& iter)
{
    typedef typename mtl::Collection<Vector>::value_type 	Scalar;
    typedef typename mtl::Collection<Vector>::size_type 	size_type;
    const Scalar     zero= math::zero(b[0]); 
    Scalar	     tmp;
    size_type nrows= num_rows(A);
    mtl::dense_vector<Scalar>   c(nrows-1, zero), d(nrows, zero);
    
    if (nrows != num_cols(A)) throw mtl::logic_error("need quadratic input matrix");
    
    //forward modification
    if (A[0][0] != zero) {
	c[0]= A[0][1]/A[0][0];
	d[0]= b[0]/A[0][0];
    } else 
	throw mtl::logic_error("no tridiagonal matrix ");
         
    for (size_type i= 1; i < nrows-1; i++) {
	tmp = A[i][i]-c[i-1]*A[i][i-1];
	if (tmp != zero) {
	    c[i]= A[i][i+1]/tmp;
	    d[i]= (b[i]-d[i-1]*A[i][i-1])/tmp;
	} else
	    throw mtl::logic_error("no tridiagonal matrix");
    }
    //treated separately last entry  //size_type is unsigned
    d[nrows-1]= (b[nrows-1]-d[nrows-2]*A[nrows-1][nrows-2])/(A[nrows-1][nrows-1]-c[nrows-2]*A[nrows-1][nrows-2]);
    
    //backward insertion
    x[nrows-1]= d[nrows-1];
    for (size_type i= nrows-2; i >0 ; i--) {
	x[i]= d[i]-c[i]*x[i+1];
	++iter;
    }
    x[0]= d[0]-c[0]*x[1];
    ++iter;
    return iter;    //relativ tol and Convergence??? algo is linear
    
}
} // namespace itl

#endif // ITL_THOMAS_INCLUDE
