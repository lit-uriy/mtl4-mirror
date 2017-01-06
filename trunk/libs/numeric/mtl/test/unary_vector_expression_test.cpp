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

template <typename Value>
struct is_int_vec
  : false_type
{};

template <>
struct is_int_vec<mtl::vec::dense_vector<int> >
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
    
    // Inverse trigonometric functions    
    v = acos(u);
    std::cout << "acos(u) = " << v << "\n";
    if (std::abs(v[0] - M_PI/2.0) > 0.0001)
        throw "acos(0) should be pi/2.";
        
    v = asin(u);
    std::cout << "asin(u) = " << v << "\n";
    if (std::abs(v[4] - M_PI/2.0) > 0.0001)
        throw "asin(1) should be pi/2.";
        
    v = atan(u);
    std::cout << "atan(u) = " << v << "\n";
    if (std::abs(v[4] - M_PI/4.0) > 0.0001)
        throw "atan(1) should be pi/4.";                
        
    for (int i = 0; i < 5; ++i)
        u[i] = 1. + i / 2.0;
    std::cout << "u = " << u << "\n";
    v = acosh(u);
    std::cout << "acosh(u) = " << v << "\n";
    if (std::abs(v[2] - 1.31696)  > 0.0001)
        throw "acosh(2) should be about 1.31696.";
        
    v = asinh(u);
    std::cout << "asinh(u) = " << v << "\n";
    if (std::abs(v[2] - 1.44363)  > 0.0001)
        throw "asinh(2) should be about 1.44363.";
        
    for (int i = 0; i < 5; ++i)
        u[i] = -0.8 + i * 0.4;
    std::cout << "u = " << u << "\n";
    v = atanh(u);
    std::cout << "atanh(u) = " << v << "\n";
    if (std::abs(v[4] - 1.098612289)  > 0.0001)
        throw "atanh(0.8) should be about 1.098612289.";
       
}

template <typename Vector>
void non_int_test(Vector&, const char*, true_type)
{
}

template <typename Vector>
void non_int_test(Vector& u, const char* name, false_type)
{
    Vector v(5);
    // Trigonometric functions
    for (int i = 0; i < 5; ++i)
        u[i] = i * M_PI / 8.0;
    std::cout << name << ": u = " << u << "\n";
    
    v = acos(u);
    std::cout << "cos(u) = " << v << "\n";
    if (std::abs(v[2] - 0.667457) > 0.0001)
        throw "cos(pi/4) should be 0.667457.";
        
    v = sin(u);
    std::cout << "sin(u) = " << v << "\n";
    if (std::abs(v[4] - 1.0) > 0.0001)
        throw "asin(pi/2) should be 1.";
        
    v = tan(u);
    std::cout << "tan(u) = " << v << "\n";
    if (std::abs(v[2] - 1) > 0.0001)
        throw "tan(pi/4) should be 1.";                
        
    v = cosh(u);
    std::cout << "acosh(u) = " << v << "\n";
    if (std::abs(v[4] - 2.50918)  > 0.0001)
        throw "acosh(pi/2) should be about 2.50918.";
    
    v = sinh(u);
    std::cout << "sinh(u) = " << v << "\n";
    if (std::abs(v[4] - 2.3013)  > 0.001)
        throw "sinh(pi/2) should be about 2.3013.";
    
    v = tanh(u);
    std::cout << "tanh(u) = " << v << "\n";
    if (std::abs(v[4] - 0.917152)  > 0.0001)
        throw "tanh(pi/2) should be about 0.917152.";

}  


template <typename Vector>
void all_test(Vector& u, const char* name)
{
    iota(u, -2);
    Vector v(5);

    std::cout << name << ": u = " << u << "\n";
    v = abs(u);
    std::cout << "abs(u) = " << v << "\n";
    MTL_THROW_IF(v[0] != 2.0, mtl::runtime_error("wrong"));
    MTL_THROW_IF(v[4] != 2.0, mtl::runtime_error("wrong"));
    
}

    
template <typename Vector>
void test(const char* name)
{
    Vector u(5);
    all_test(u, name);
    non_int_test(u, name, is_complex<Vector>());
    non_complex_test(u, name, is_complex<Vector>());
    
    std::cout << "\n\n";
}
 
 
 
 
int main(int, char**)
{
    using mtl::vec::parameters;
    using namespace mtl;

    std::cout << "Testing vector operations\n";

    test<dense_vector<float> >("test float");
    test<dense_vector<double> >("test double");
    test<dense_vector<std::complex<double> > >("test complex<double>");
    test<dense_vector<float, parameters<row_major> > >("test float in row vector");
    
    return 0;
}
