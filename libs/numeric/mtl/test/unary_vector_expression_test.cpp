// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

#include <iostream>
#include <complex>

// #include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/operations.hpp>

// using namespace std;

struct true_type {};
struct false_type {};

template <typename Vector>
struct is_complex
  : false_type
{};

template <typename Value>
struct is_complex<mtl::vec::dense_vector<std::complex<Value> > >
  : true_type
{};


template <typename Vector>
void non_complex_test(Vector&, const char*, true_type)
{
}

template <typename Vector>
void non_complex_test(Vector& u, const char* name, false_type)
{    
    Vector v(5);

    for (int i = 0; i < 5; ++i)
        u[i] = i / 4.0;
    std::cout << "Non-complex-" << name << ": u = " << u << "\n";
    
    v = acos(u);
    std::cout << "acos(u) = " << v << "\n";
    if (std::abs(v[0] - M_PI/2.0) > 0.0001)
        throw "acos(0) should be pi/2.";
}



template <typename Vector>
void test(Vector& u, const char* name)
{
    // u= 3.0; v= 4.0; w= 5.0;
    iota(u, -2);
    Vector v(5);

    std::cout << name << ": u = " << u << "\n";
    v = abs(u);
    std::cout << "abs(u) = " << v << "\n";
    MTL_THROW_IF(v[0] != 2.0, mtl::runtime_error("wrong"));
    MTL_THROW_IF(v[4] != 2.0, mtl::runtime_error("wrong"));
    
    
    
    non_complex_test(u, name, is_complex<Vector>());
    
    std::cout << "\n\n";
}
    
    
    
int main(int, char**)
{
    using mtl::vec::parameters;
    using namespace mtl;

    dense_vector<float>   u(5);
    dense_vector<double>  x(5);
    dense_vector<std::complex<double> >  xc(5);
    dense_vector<float, parameters<row_major> >   ur(5);

    std::cout << "Testing vector operations\n";

    test(u, "test float");
    test(x, "test double");
    test(xc, "test complex<double>");
    test(ur, "test float in row vector");
    
    return 0;
}
