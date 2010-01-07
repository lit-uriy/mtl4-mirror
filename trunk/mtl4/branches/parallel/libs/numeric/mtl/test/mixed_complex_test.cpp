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
#include <string>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;

// Function is instantiated 108 times
template <typename T, typename U>
void inline test3(string type1, T x, string type2, U y)
{
    cout << type1 << " @ " << type2 << '\n';
    // cout << "As typeid: " << typeid(x).name() << " @ " << typeid(y).name() << '\n';
    // cout << x << " @ " << y << '\n';

    cout << x << " + " << y << " = " << x + y << '\n';  
    if (x + y != 8) throw "Result of addition must be 8.";  // Integer numbers less then 17 should not have rounding errors

    cout << x << " - " << y << " = " << x - y << '\n';
    if (x - y != 4) throw "Result of subtraction must be 4.";

    cout << x << " * " << y << " = " << x * y << '\n';
    if (x * y != 12) throw "Result of subtraction must be 12.";

    cout << x << " / " << y << " = " << x / y << '\n';
    if (x / y != 3) throw "Result of subtraction must be 3.";

    cout << '\n';
}

template <typename T, typename U>
void inline test2(const char* type1, T x, const char* type2, U y)
{
    cout << type1 << " @ " << type2 << '\n';
    cout << "different_non_complex<T, U> is " << (mtl::traits::different_non_complex<T, U>::value ? "true (use my extension)\n " : "false (use standard)\n") << "----\n";
    string ctype1(string("complex<") + type1 + ">"), ctype2(string("complex<") + type2 + ">");
    std::complex<T> cx(x);
    std::complex<U> cy(y);

    test3(type1, x, ctype2, cy);
    test3(ctype1, cx, type2, y);
    test3(ctype1, cx, ctype2, cy);
}

template <typename T>
void inline test(const char* type1, T x)
{
    test2(type1, x, "int", 2);
    test2(type1, x, "long", 2l);
    test2(type1, x, "unsigned", 2u);
    test2(type1, x, "float", 2.f);
    test2(type1, x, "double", 2.);
    test2(type1, x, "long double", 2.l);
}


int test_main(int argc, char* argv[])
{
    using namespace mtl;

    test("int", 6);
    test("long", 6l);
    test("unsigned", 6u);
    test("float", 6.f);
    test("double", 6.);
    test("long double", 6.l);

    return 0;
}
