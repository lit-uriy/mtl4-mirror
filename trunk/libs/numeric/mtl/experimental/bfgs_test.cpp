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

#include <iostream>
#include <utility>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace std;  
   
struct f_test
{
    template <typename Vector>
    typename mtl::Collection<Vector>::value_type 
    operator() (const Vector& x) const
    {
	return x[0]*x[0] + 2*x[1]*x[1] + 2*x[2]*x[2];
    }
};

struct grad_f_test
{
    template <typename Vector>
    Vector operator() (const Vector& x) const
    {    
	Vector tmp(size(x));
	tmp[0]= 2 * x[0];
	tmp[1]= 4 * x[1];
	tmp[2]= 4 * x[2];
	return tmp;
    }
};

template <typename Value= double>
class wolf
{
  public:
    typedef Value   value_type;

    // Defaults from Prof. Fischer's lecture
    wolf(Value delta= 0.5, Value gamma= 0.5, Value beta1= 0.25, Value beta2= 0.5)
      : delta(delta), gamma(gamma), beta1(beta1), beta2(beta2) {}

    template <typename Vector, typename F, typename Grad>
    typename mtl::Collection<Vector>::value_type 
    operator() (const Vector& x, const Vector& d, F f, Grad grad_f) 
    {
	// Star's step size
	value_type alpha= -gamma * dot(grad_f(x), d) / dot(d, d);
	mtl::dense_vector<value_type> x_k(x + alpha * d);

	while (f(x_k) > f(x) + (beta1 * alpha) * dot(grad_f(x), d) 
	       && dot(grad_f(x_k), d) < beta2 * dot(grad_f(x), d)) {	
	    alpha*= (beta1 + beta2) / 2;
	    x_k= x+ alpha * d;
	    std::cout<< "alpha_a=" << alpha << "\n";
	}
	return alpha;
    }
  private:
    Value delta, gamma, beta1, beta2; 
};

struct bfgs
{
    template <typename Matrix, typename Vector>
    void operator() (Matrix& H, const Vector& y, const Vector& s)
    {
	typedef typename mtl::Collection<Vector>::value_type value_type;
	assert(num_rows(H) == num_cols(H));

	value_type gamma= 1 / dot(y,s);
	Matrix     A(math::one(H) - gamma * s * trans(y)),
	           H2(A * H * trans(A) + gamma * s * trans(s));
	swap(H2, H); // faster than H= H2
    }
}; 

template <typename Matrix, typename Vector, typename F, typename Grad, 
	  typename Step, typename Update, typename Iter>
Vector quasi_newton(Vector& x, F f, Grad grad_f, Step step, Update update, Iter& iter) 
{    
    typedef typename mtl::Collection<Vector>::value_type value_type;
    Vector         d, y, x_k, s;
    Matrix         H(size(x), size(x));
    
    H= 1;
    while (!iter.finished(two_norm(grad_f(x)))) {
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

template <typename Vector, typename F, typename Grad, typename Step, typename Update, typename Iter>
Vector inline quasi_newton(Vector& x, F f, Grad grad_f, Step step, Update update, Iter& iter) 
{
    typedef typename mtl::Collection<Vector>::value_type value_type;
    return quasi_newton<mtl::dense2D<value_type> >(x, f, grad_f, step, update, iter);
}

int test_main(int, char**)
{
    using namespace mtl;

    mtl::dense_vector<double>       x(3, 8);
    std::cout<< "x= " << x << "\n";
    
    itl::cyclic_iteration<double> iter(0.0, 100, 0, 1e-4, 100);
    quasi_newton(x, f_test(), grad_f_test(), wolf<>(), bfgs(), iter);

    std::cout<< "x= " << x << "\n";
    std::cout<< "grad_f(x)= " << grad_f_test()(x) << "\n";
    if (two_norm(x) > 10 * iter.atol())
	throw "x should be 0.";

    return 0;
}
 














