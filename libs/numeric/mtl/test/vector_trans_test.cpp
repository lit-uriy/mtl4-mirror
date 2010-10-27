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
#include <complex>
#include <vector>

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>


template <typename Vector>
void test(const char* name, Vector& v)
{
    v= 3., 4., 6.;
    std::cout << name << ": v is " << v << ", typeid is " << typeid(v).name() << "\n";
    std::cout << "trans(v) is " << trans(v) << ", typeid is " << typeid(trans(v)).name() << "\n";
}

template <typename Vector>
void test2(const char* name, const Vector& v)
{
    std::cout << name << ": v is " << v << ", typeid is " << typeid(v).name() << "\n";
    std::cout << "trans(v) is " << trans(v) << ", typeid is " << typeid(trans(v)).name() << "\n";
}


int test_main(int, char**)
{

    typedef mtl::vector::fixed::dimension<3> fsize;
    mtl::dense_vector<float, mtl::vector::parameters<mtl::row_major, fsize, true> >     rf;
    mtl::dense_vector<float, mtl::vector::parameters<mtl::col_major, fsize, true> >     cf;

    mtl::dense_vector<float, mtl::vector::parameters<mtl::row_major> >                  rd(3);
    mtl::dense_vector<float>                                                            cd(3);

    mtl::dense_vector<std::complex<double> >                                            rdc(3);

    mtl::dense2D<float>  A(3, 3);
    A= 2, 3, 4,
       3, 4, 6,
       7, 6, 9;

    test("Row vector fixed size", rf);
    test("Column vector fixed size", cf);
    test("Row vector", rd);
    test("Column vector", cd);
    test("Row vector complex", rdc);
    test2("Matrix row", A[mtl::iall][1]);
    test2("Matrix column", A[1][mtl::iall]);

    return 0;
}
