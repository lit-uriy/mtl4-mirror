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

#include <iostream>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/vector/parameter.hpp>
#include <boost/numeric/mtl/operation/operators.hpp>
#include <boost/numeric/mtl/operation/norms.hpp>
#include <boost/numeric/mtl/operation/sum.hpp>
#include <boost/numeric/mtl/operation/product.hpp>


using namespace std;  
    

template <typename Vector>
void test(Vector& v, const char* name)
{
    typedef typename mtl::Collection<Vector>::value_type value_type;
    typedef typename mtl::Collection<Vector>::size_type  size_type;

    using mtl::sum; using mtl::product; using mtl::one_norm;

    for (size_type i= 0; i < size(v); i++)
	v[i]= value_type(double(i+1) * pow(-1.0, int(i))); // Amb. in MSVC

    std::cout << "\n" << name << "  --- v = " << v; std::cout.flush();

    Vector w(v + v), u;
    std::cout << "w= v + v = " << w << "\n"; 

    u= w - v;
    std::cout << "u= w - v = " << u << "\n"; 

    u= -v;
    std::cout << "u= -v = " << u << "\n"; 
    
}
 

int test_main(int argc, char* argv[])
{
    using namespace mtl;
    using mtl::vector::parameters;

    dense_vector<float>   u(5);
    dense_vector<double>  x(5);
    dense_vector<std::complex<double> >  xc(5);

    std::cout << "Testing vector operations\n";

    test(u, "test float");
    test(x, "test double");

    test(xc, "test complex<double>");

    dense_vector<float, parameters<row_major> >   ur(5);
    test(ur, "test float in row vector");

    return 0;
}
 














