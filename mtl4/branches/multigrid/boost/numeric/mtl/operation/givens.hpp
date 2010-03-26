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

// With contributions from Cornelius Steinhardt

#ifndef MTL_MATRIX_GIVENS_INCLUDE
#define MTL_MATRIX_GIVENS_INCLUDE

#include <cmath>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/irange.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>


namespace mtl { namespace matrix {

/// Given's transformator
template <typename Matrix>
class givens
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;

  public:
    /// Constructor takes %matrix \p H to be transformed and the rotation parameters \p a and \p b
    givens(Matrix& H, value_type a, value_type b) : H(H)
    {
	using std::abs;
	value_type zero= math::zero(a), one= math::one(b), t;

	if ( b == zero ) {
	    c= one; d= zero;
	} else if ( abs(b) > abs(a) ) {
	    t= -a/b;
	    d= one/sqrt(one + t*t);
	    c= d*t;
	} else {
	    t= -b/a;
	    c= one/sqrt(one + t*t);
	    d= c*t;
	}
  }

    /// Given's transformation of \p H with \p G regarding column \p k
    Matrix& trafo(const Matrix& G, size_type k)
    {
	    irange r(k,k+2);
	    // trans(H[r][ind])*= G; H[ind][r]*= G; // most compact form but does not work yet
	    
	    Matrix col_block(H[r][iall]), col_perm(trans(G) * col_block);
	    H[r][iall]= col_perm; 
	    Matrix row_perm(H[iall][r] * G);
	    H[iall][r]= row_perm;

	    return H;
    }

    /// Given's transformation of \p H regarding column \p k
    Matrix& trafo(size_type k)
    {
	Matrix G(2, 2);
	G= c, d,
	  -d, c;
	return trafo(G, k);
    }

  private:
    Matrix&    H;
    value_type c, d;
};

}// namespace matrix


namespace vector {

/// Given's transformator on %vector (swap a*line(k) with b*line(k+1) )
template <typename Vector>
class givens
{
    typedef typename Collection<Vector>::value_type   value_type;
    typedef typename Collection<Vector>::size_type    size_type;

  public:
    /// Constructor takes %vector \p H to be transformed and the rotation parameters \p a and \p b
    givens(Vector& H, value_type a, value_type b) : H(H), a(a), b(b)
    {  }

    /// Given's transformation of \p H with \p G regarding column \p k
    Vector& trafo(size_type k)
    {
	    value_type w1(0), w2(0);
	    w1= a*H[k]-b*H[k+1]; //given's rotation on solution
            w2= b*H[k]+a*H[k+1]; //rotation on vector
            H[k]= w1;
            H[k+1]= w2;

	    return H;
    }

  private:
    Vector&    H;
    value_type a, b;
};

}// namespace vector



} // namespace mtl

#endif // MTL_MATRIX_GIVENS_INCLUDE
