// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschrÃ¤nkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.
 
// #define MTL_VERBOSE_TEST
#define MTL_HAS_STD_OUTPUT_OPERATOR

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/matrix/ell_matrix.hpp>

template <typename Matrix>
inline void fill_matrix(Matrix& A)
{
    mtl::matrix::inserter<Matrix> ins(A, 3);
    ins[0][0] << 2;
    ins[0][1] << 9;
    ins[1][1] << 1;
    ins[1][2] << 5;
    ins[1][3] << 5;
    ins[1][4] << 1;
    ins[2][2] << 6;
    ins[2][3] << 9;
    ins[3][2] << 2;
    ins[3][3] << 4;
    ins[4][0] << 7;
    ins[4][4] << 3;
}

int main(int, char**)
{
    using namespace mtl;
    using mtl::io::tout;
    typedef mtl::matrix::ell_matrix<double>   matrix_type;
    typedef matrix_type::size_type            size_type;
    matrix_type   A(5, 5);

    fill_matrix(A);

    tout << "A (internal)\n";
    A.print_internal(tout);

    tout << "A[2][3] = " << A[2][3] << '\n';
    tout << "A[2][4] = " << A[2][4] << '\n';
    tout << "A[2][0] = " << A[2][0] << '\n';

    MTL_THROW_IF(A[2][3] != 9.0, unexpected_result())
    MTL_THROW_IF(A[2][4] != 0.0, unexpected_result())
    MTL_THROW_IF(A[2][0] != 0.0, unexpected_result())

    tout << "A =\n" << A;
    tout << "nnz = " << A.nnz() << std::endl;

    mtl::matrix::compressed2D<double> B(5, 5);
    fill_matrix(B);
    tout << "B =\n" << B;
    MTL_THROW_IF(A.nnz() != B.nnz(), unexpected_result())
    
    mtl::dense_vector<double> res(5), res2(5), x(5);
    iota(x, 1);
    res2= B * x;
    tout << "B * x = " << res2 << '\n';
    
    res= A * x;
    tout << "A * x =\n" << res << '\n';

    res2-= res;
    MTL_THROW_IF(two_norm(res2) > 0.001, unexpected_result())
	
    return 0;
}
