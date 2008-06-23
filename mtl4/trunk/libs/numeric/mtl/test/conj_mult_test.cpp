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


using namespace std;
using namespace mtl;

typedef std::complex<double>      cdouble;

template <typename Matrix>
void test(Matrix& A, const char* name)
{
    const unsigned                    xd= 2, yd= 5, n= xd * yd;
    A.change_dim(n, n);
    matrix::laplacian_setup(A, xd, yd); 

    A*= cdouble(1, -1);
    std::cout << name << "\nconj(A) is\n" << with_format(conj(A), 7, 1) << "\n";

    dense_vector<cdouble> x(n),Ax(n);
    x=cdouble(1,2);
    
    Ax=conj(A) * x;
    std::cout << "conj(A) * x is " << Ax << "\n";
    
    Ax=trans(A) * x;
    std::cout << "trans(A) * x is " << Ax << "\n";

    Ax=hermitian(A) * x;
    std::cout << "hermitian(A) * x is " << Ax << "\n";
}


int test_main(int argc, char* argv[])
{
    compressed2D<cdouble>             crc;

    test(crc, "Compressed row major complex");

    return 0;
}
