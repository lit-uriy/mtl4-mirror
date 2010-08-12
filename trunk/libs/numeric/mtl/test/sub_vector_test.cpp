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
    typedef typename mtl::Collection<Vector>::value_type T;
    using std::abs; using std::cout;
    std::cout << "\n" << name << "\n";

    for (unsigned i= 0; i < size(v); ++i)
	v[i]= f(T(i));


    Vector w(sub_vector(v, 2, 4));
    w[1]= f(T(8));
    cout << "w == " << w << "\n";

    if (w[0] != f(T(2))) throw "Wrong value in w";
    if (size(w) != 2)    throw "Wrong size of w";

    if (v[3] != f(T(8))) throw "Cannot change v via w (correctly)";

    Vector u(sub_vector(v, 2, 7));
    cout << "u == " << u << "\n";
    cout << "v == " << v << "\n";

    if (u[0] != f(T(2))) throw "Wrong value in u";
    if (size(u) != sz-2) throw "Wrong size of u";

    if (size(sub_vector(v, 2, 2)) != 0) throw "Problem returning empty vector (2, 2)";
    if (size(sub_vector(v, 4, 2)) != 0) throw "Problem returning empty vector (4, 2)";
    if (size(sub_vector(v, 8, 9)) != 0) throw "Problem returning empty vector (8, 9)";

#if 0
    const Vector vc(v);
    cout << "vc == " << vc << "\n";

    Vector wc(sub_vector(vc, 2, 4));
    const Vector cwc(sub_vector(vc, 2, 4));    
    cout << "wc == " << wc << "\n";

    if (wc[0] != f(T(2))) throw "Wrong value in wc";
    if (size(wc) != 2)    throw "Wrong size of wc";    

    wc[1]= f(T(5));
    cout << "vc == " << vc << "\n";
#endif
}



int test_main(int argc, char* argv[])
{
    mtl::dense_vector<float>                                                 cf(sz, 1.0);
    mtl::dense_vector<double>                                                cd(sz, 1.0);
    mtl::dense_vector<std::complex<double> >                                 cc(sz, 1.0);
    mtl::dense_vector<float, mtl::vector::parameters<mtl::row_major> >       rf(sz, 1.0);

    test(cf, "dense_vector<float>");
    test(cd, "dense_vector<double>");
    test(cc, "dense_vector<std::complex<double> >");
    test(rf, "dense_vector<float, vector::parameters<row_major> >");

    return 0;
}
