// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


// Everything in the test is double
// How to test sparse generically? 

using namespace std;
using namespace mtl;
	

// Return a vector with move semantics
// Return also the address of the first entry to be sure that it is really moved
template <typename Vector>
Vector f(const Vector&, double*& a00)
{
    Vector v(3);
    v= 5.0;
    a00= &v.data[0];
    return v;
}

template <typename Vector>
void print(const Vector& vector, const double* p)
{
    cout << "Data was " << ((const double*) &vector.data[0] == p ? "moved.\n" : "copied.\n");
}

template <typename Vector>
void test(const Vector&, const char* text)
{
    cout << '\n' << text << '\n';

    double *p;
    Vector v(3);
    v= 0.0;
   
    cout << "v= f(v, p);\n";
    v= f(v, p);
    print(v, p);

    if (v.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&v.data[0] != p) 
	throw "Vector is not moved but copied!";

    cout << "Vector w= f(v, p);\n";
    Vector w= f(v, p);
    print(w, p);

    if (w.data[0] != 5.0) 
	throw "Wrong value moving, should be 5.0!";
    if (&w.data[0] != p) 
	throw "Vector is not moved but copied!";

    // This type is guarateed to be different to f's return type
    // In this case the vector MUST be copied
    dense_vector<float>    x(3);

    cout << "x= f(v, p);  // x and v have different types\n";
    x= f(v, p);
    print(x, p);

    if (x.data[0] != 5.0) 
	throw "Wrong value trying to move, should be 5.0!";
    if (&x.data[0] == (float*) p) 
	throw "Vector must be copied not moved!";

    // Other vector type, in this case the vector MUST be copied
    dense_vector<float>    y(v);

    cout << "y(v);  // x and v have different types\n";
    print(y, &v.data[0]);

    if (y.data[0] != 5.0) 
	throw "Wrong value in copy constructor, should be 5.0!";
    if (&y.data[0] == (float*) &v.data[0]) 
	throw "Vector must be copied not moved!";


}




int test_main(int argc, char* argv[])
{
    dense_vector<double>                                 dr(3);

    test(dr, "Dense vector");

    return 0;
}
