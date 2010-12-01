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


// #include <boost/numeric/mtl/mtl.hpp>
// #include <boost/numeric/itl/itl.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/operation/two_norm.hpp>
#include <boost/numeric/itl/iteration/cyclic_iteration.hpp>

#include <boost/numeric/itl/minimization/quasi_newton.hpp>
#include <boost/numeric/itl/stepper/wolf.hpp>
#include <boost/numeric/itl/updater/bfgs.hpp>


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


int test_main(int, char**)
{
    using namespace mtl;

    mtl::dense_vector<double>       x(3, 8);
    std::cout<< "x= " << x << "\n";
        
    itl::cyclic_iteration<double> iter(grad_f_test()(x), 1000, 0, 1e-4, 100);
    quasi_newton(x, f_test(), grad_f_test(), itl::wolf<>(), itl::bfgs(), iter);
    iter.error_code();    

    std::cout<< "x= " << x << "\n";
    std::cout<< "grad_f(x)= " << grad_f_test()(x) << "\n";
    if (two_norm(x) > 10 * iter.atol())
	throw "x should be 0.";

    return 0;
}
 














