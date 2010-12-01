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

#ifndef ITL_QUASI_NEWTON_INCLUDE
#define ITL_QUASI_NEWTON_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/matrix/operators.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>

namespace itl {

/// Quasi-Newton method
template <typename Matrix, typename Vector, typename F, typename Grad, 
	  typename Step, typename Update, typename Iter>
Vector quasi_newton(Vector& x, F f, Grad grad_f, Step step, Update update, Iter& iter) 
{    
    typedef typename mtl::Collection<Vector>::value_type value_type;
    Vector         d, y, x_k, s;
    Matrix         H(size(x), size(x));
    
    H= 1;
    for (; !iter.finished(two_norm(grad_f(x))); ++iter) {
	d= H * -grad_f(x);                               
	value_type alpha= step(x, d, f, grad_f);
	x_k= x + alpha * d;
	s= alpha * d;
	y= grad_f(x_k) - grad_f(x);
	update(H, y, s);                               
	x= x_k;
    }
    return x;
}

/// Quasi-Newton method
template <typename Vector, typename F, typename Grad, typename Step, typename Update, typename Iter>
Vector inline quasi_newton(Vector& x, F f, Grad grad_f, Step step, Update update, Iter& iter) 
{
    typedef typename mtl::Collection<Vector>::value_type value_type;
    return quasi_newton<mtl::dense2D<value_type> >(x, f, grad_f, step, update, iter);
}


} // namespace itl

#endif // ITL_QUASI_NEWTON_INCLUDE
