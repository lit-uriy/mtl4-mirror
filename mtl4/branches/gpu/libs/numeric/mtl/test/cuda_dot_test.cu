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
#include <complex>
#include <cmath>

#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <boost/numeric/mtl/cuda/dot.cu>


template <typename VectorU>
void test(VectorU& u, VectorU& v, const char* name)
{
    //using mtl::vector::dot;
    typedef typename mtl::Collection<VectorU>::size_type  size_type;
    for (size_type i= 0; i < size(v); i++)
	u[i]= i+1, v[i]= i+1;

    std::cout << name << "\n dot(u, v) = " << dot(u, v) << "\n"; std::cout.flush();
    if (std::abs(dot(u, v) - 285.0) > 0.01) throw "dot product wrong";
}
 



int main( int argc, char** argv)
{
    const int size= 9;

    mtl::cuda::vector<int>     i(size), j(size);
    mtl::cuda::vector<float>   u(size), v(size), w(size);
    mtl::cuda::vector<double>  x(size), y(size), z(size);
    mtl::cuda::vector<std::complex<double> >  xc(size), yc(size), zc(size);

    test(i, j, "test int");
    test(u, v, "test float");
    test(x, y, "test double");
    // test(xc, yc, "test complex<double>");


//     mtl::cuda::vector_cuda<float, parameters<mtl::row_major> >   ur(size), vr(size), wr(size);
//     test(ur, vr, "test float in row vector");

    return 0;
}
