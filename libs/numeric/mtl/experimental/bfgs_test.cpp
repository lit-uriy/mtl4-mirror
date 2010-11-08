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

struct f_test
{
    template <typename Vector>
    typename mtl::Collection<Vector>::value_type 
    operator() (const Vector& x) const
    {
	return x[0]*x[0] + 2*x[1]*x[1] + 2*x[2]*x[2];
    }
};

struct armijo
{
    template <typename Vector, typename F, typename Grad>
    typename mtl::Collection<Vector>::value_type 
    operator() (const Vector& x, const Vector& d, F f, Grad grad_f) 
    {
	typedef typename mtl::Collection<Vector>::value_type value_type;
	value_type delta= 0.5, gamma= 0.5, beta1= 0.25, beta2= 0.5;  //feste Werte

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
};

struct bfgs
{
    template <typename Matrix, typename Vector>
    void operator() (Matrix& H, const Vector& y, const Vector& s)
    {
	typedef typename mtl::Collection<Vector>::value_type value_type;
	assert(num_rows(H) == num_cols(H));

	value_type gamma= 1 / dot(y,s);
	Matrix     A(math::one(H) - gamma * y * trans(s)),
	           H2(trans(A) * H * A + gamma * s * trans(s));
	swap(H2, H); // faster than H= H2
    }
}; 

template <typename Vector, typename F, typename Grad, typename Step, typename Update>
Vector quasi_newton(Vector& x, F f, Grad grad_f, Step step, Update update, double tol) 
{    
    typedef typename mtl::Collection<Vector>::value_type value_type;
    Vector                    d_k, y_k, x_k, s_k;
    mtl::dense2D<value_type>  H(mtl::matrix::identity<value_type>(size(x))); 
   
    while (two_norm(grad_f(x)) > tol) {
	d_k= H * -grad_f(x);                               
	value_type alpha= step(x, d_k, f, grad_f);
	x_k= x + alpha * d_k;
	s_k= alpha * d_k;
	y_k= grad_f(x_k) - grad_f(x);
	update(H, y_k, s_k);                               
	x= x_k;
    }
    return x;
}

int test_main(int, char**)
{
    using namespace mtl;

    mtl::dense_vector<double>       x0(3, 8);
    double tol= 1e-4;
    std::cout<< "x0= " << x0 << "\n";
    
    quasi_newton(x0, f_test(), grad_f_test(), armijo(), bfgs(), tol);
    std::cout<< "x0= " << x0 << "\n";
    std::cout<< "grad_f(x0)= " << grad_f_test()(x0) << "\n";
    

    return 0;
}
 














