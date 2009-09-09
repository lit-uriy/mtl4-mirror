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
#include <boost/numeric/mtl/operation/householder.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>


namespace mtl { namespace matrix {


template <typename Matrix>
class givens
{
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    
  public:
    givens(Matrix& H, value_type a, value_type b) : H(H)
    {
	using std::abs;
	value_type zero= math::zero(a), one= math::one(b), t;

	if ( b == zero ) 
	    std::make_pair(one, zero);

	if ( abs(b) > abs(a) ) {
	    t= -a/b;
	    d= one/sqrt(one + t*t);
	    c= d*t;
	} else {
	    t= -b/a;
	    c= one/sqrt(one + t*t);
	    d= c*t;
	}
    }

    Matrix& trafo(const Matrix& G, size_type k)
    {
	size_type        ncols = num_cols(H), nrows = num_rows(H);
	value_type       zero= math::zero(H[0][0]), one= math::one(H[0][0]);
	//Matrix           Tmp(H), Tmp1(H);

	irange r(k,k+2), ind;
	//only important components of givensrotation
	if (k > 0 && k < nrows-2 && nrows >= 3) 
	    ind.set(k-1,k+2);
	else if (k == nrows-2 && nrows >= 3) 
	    ind.set(nrows-3,nrows-1);
	else if (k == 0 && nrows >= 3) 
	    ind.set(0,2);
	else if ( nrows == 1 )
	    return H;
	else
	    MTL_THROW(logic_error("Cornelius claims this never happens"));

	// H[r][ind]= trans(G) * H[r][ind] (or trans(H[r][ind])*= G;)
	Matrix Tmp(H[r][ind]);
	H[r][ind]= trans(G) * Tmp;

	// H[ind][r]*= G;
	Matrix Tmp1(H[ind][r] * G);
	H[ind][r]= Tmp1;

#if 0
	//transformation of H
	H[r][ind] = trans(G) * Tmp[r][ind];
	Tmp= H;
	Tmp1=zero;
	Tmp1[ind][r] = Tmp[ind][r]*G;
	H[ind][r]= Tmp1[ind][r];
#endif

	return H;
    }

    Matrix& trafo(size_type k)
    {
	Matrix G(2, 2);
	G= c, d,
	  -d, c;
	return trafo(G, k);
    }
    
  private:
    Matrix& H;
    value_type c, d;
};




#if 0
/// Parameters of Given's rotation
template <typename Value>
std::pair<Value, Value> inline givens_param(Value a, Value b)
{
    using std::abs;
    const Value zero= math::zero(a), one= math::one(b), t, d, c;

    if ( b == zero ) 
	std::make_pair(one, zero);

    if ( abs(b) > abs(a) ) {
	t= -a/b;
	d= one/sqrt(one + t*t);
	c= d*t;
    } else {
	t= -b/a;
	c= one/sqrt(one + t*t);
	d= c*t;
    }
    return std::make_pair(c,d);
}


/// Given's transformation of \p H with \p G regarding column \p k
template <typename Matrix>
Matrix& inline givens_trafo(Matrix& H, const Matrix& G, typename Collection<Matrix>::size_type k)
{
    //Evaluation of A= G'*A*G for Tridiagonal A and Givensrotaion G(k,k+1)
    typedef typename Collection<Matrix>::value_type   value_type;
    typedef typename Collection<Matrix>::size_type    size_type;
    size_type        ncols = num_cols(H), nrows = num_rows(H);
    value_type       zero= math::zero(H[0][0]), one= math::one(H[0][0]);
    Matrix           Tmp(H), Tmp1(H);

    irange r(k,k+2), ind;
    //only important components of givensrotation
    if ((k > 0) && (k < nrows-2) && (nrows >= 3)) {
        irange ind(k-1,k+2);
    }else if ((k == nrows-2) && (nrows >= 3)) {
        irange ind(nrows-3,nrows-1);
    }else if ((k == 0) && (nrows >= 3)) {
        irange ind(0,2);
    }else if ( nrows == 1 ){
        irange ind(0,1);
    }

    //transformation of H
    H[r][ind] = trans(G)*Tmp[r][ind];
    Tmp=H;
    Tmp1=zero;
    Tmp1[ind][r] = Tmp[ind][r]*G;
    H[ind][r]= Tmp1[ind][r];
    return H;
}

/// Given's transformation of \p H with \p G regarding column \p k
template <typename Matrix>
Matrix& inline givens_trafo(Matrix& H, typename Collection<Matrix>::value_type const& c
			    typename Collection<Matrix>::value_type const& d, 
			    typename Collection<Matrix>::size_type k)
{
    Matrix G(2, 2);
    G= c, d,
      -d, c;
    return givens_trafo(H, G, k);
}


#endif

}} // namespace mtl::matrix

#endif // MTL_MATRIX_GIVENS_INCLUDE

