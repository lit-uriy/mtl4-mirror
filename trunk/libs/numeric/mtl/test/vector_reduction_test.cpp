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
    using mtl::sum; using mtl::product; using mtl::one_norm;

    for (unsigned i= 0; i < size(v); i++)
	v[i]= value_type(double(i+1) * pow(-1.0, int(i))); // MSVC considers pow)(, i) ambiguous 

    std::cout << "\n" << name << "  --- v = " << v; std::cout.flush();

    std::cout << "\none_norm(v) = " << one_norm(v) << "\n"; std::cout.flush();
    if (one_norm(v) != 15.0) throw "one_norm wrong";

    std::cout << "one_norm<4>(v) = " << one_norm<4>(v) << "\n"; std::cout.flush();
    if (one_norm<4>(v) != 15.0) throw "one_norm<4> wrong";

    std::cout << "two_norm(v) = " << two_norm(v) << "\n"; std::cout.flush();
    if (two_norm(v) < 7.4161 || two_norm(v) > 7.4162) throw "two_norm wrong";

    std::cout << "infinity_norm(v) = " << infinity_norm(v) << "\n"; std::cout.flush();
    if (infinity_norm(v) != 5.0) throw "infinity_norm wrong";

    std::cout << "sum(v) = " << sum(v) << "\n"; std::cout.flush();
    if (sum(v) != 3.0) throw "sum wrong";

    std::cout << "sum<3>(v) = " << sum<3>(v) << "\n"; std::cout.flush();
    if (sum<3>(v) != 3.0) throw "sum<3> wrong";

    std::cout << "product(v) = " << product(v) << "\n"; std::cout.flush();
    if (product(v) != 120.0) throw "product wrong";

    std::cout << "product<6>(v) = " << product<6>(v) << "\n"; std::cout.flush();
    if (product<6>(v) != 120.0) throw "product<6> wrong";
}
 

int test_main(int, char**)
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
 














