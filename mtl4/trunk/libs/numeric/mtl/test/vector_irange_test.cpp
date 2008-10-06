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
#include <complex>
#include <vector>

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

using namespace mtl;

const unsigned sz= 5;

inline float f(float x) { return x; }
inline double f(double x) { return x; }

inline std::complex<double> f(std::complex<double> x) 
{ 
    return std::complex<double>(real(x), real(x)+1.0); 
}


template <typename Vector>
void test(Vector& v, const char* name)
{
    typedef typename Collection<Vector>::value_type T;
    using std::abs; using std::cout;
    std::cout << "\n" << name << "\n";

    for (unsigned i= 0; i < size(v); ++i)
	v[i]= f(T(i));


    Vector w(v[irange(2, 4)]);
    w[1]= f(T(8));
    cout << "w == " << w << "\n";

    if (w[0] != f(T(2))) throw "Wrong value in w";
    if (size(w) != 2)    throw "Wrong size of w";

    if (v[3] != f(T(8))) throw "Cannot change v via w (correctly)";

    Vector u( v[irange(2, imax)] );
    cout << "u == " << u << "\n";
    cout << "v == " << v << "\n";

    if (u[0] != f(T(2))) throw "Wrong value in u";
    if (size(u) != sz-2) throw "Wrong size of u";

    cout << "v[irange(2, 4)] == " << v[irange(2, 4)] << "\n";
}



int test_main(int argc, char* argv[])
{
    dense_vector<float>                                                 cf(sz, 1.0);
    dense_vector<double>                                                cd(sz, 1.0);
    dense_vector<std::complex<double> >                                 cc(sz, 1.0);
    dense_vector<float, vector::parameters<row_major> >                 rf(sz, 1.0);

    test(cf, "dense_vector<float>");
    test(cd, "dense_vector<double>");
    test(cc, "dense_vector<std::complex<double> >");
    test(rf, "dense_vector<float, vector::parameters<row_major> >");

    return 0;
}
