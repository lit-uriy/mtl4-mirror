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

#include <string>
#include <iostream>

#define MTL_VERBOSE_TEST
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/mtl/matrix/sparse_banded.hpp>


template <typename Matrix>
void laplacian_test(Matrix& A, unsigned dim1, unsigned dim2, const char* name)
{
    mtl::io::tout << "\n" << name << "\n";
    laplacian_setup(A, dim1, dim2);
    mtl::io::tout << "Laplacian A:\n" << A << std::endl;
    if (dim1 > 1 && dim2 > 1) {
	typename Matrix::value_type four(4.0), minus_one(-1.0), zero(0.0);
	MTL_THROW_IF(A[0][0] != four, mtl::runtime_error("wrong diagonal"));
	MTL_THROW_IF(A[0][1] != minus_one, mtl::runtime_error("wrong east neighbor"));
	MTL_THROW_IF(A[0][dim2] != minus_one, mtl::runtime_error("wrong south neighbor"));
	MTL_THROW_IF(dim2 > 2 && A[0][2] != zero, mtl::runtime_error("wrong zero-element"));
	MTL_THROW_IF(A[1][0] != minus_one, mtl::runtime_error("wrong west neighbor"));
	MTL_THROW_IF(A[dim2][0] != minus_one, mtl::runtime_error("wrong north neighbor"));
	MTL_THROW_IF(dim2 > 2 && A[2][0] != zero, mtl::runtime_error("wrong zero-element"));
    }
}

template <typename Matrix>
void rectangle_test(Matrix& A, const char* name)
{
    {
	mtl::matrix::inserter<Matrix> ins(A);
	int i= 1;
	unsigned nc= num_cols(A);
	for (unsigned r= 0; r < num_rows(A); r++) {
	    if (r < nc - 4) ins(r, r + 4) << i++;
	    if (r < nc) ins(r, r) << i++;
	    if (r >= 2 && r < nc + 2) ins(r, r - 2) << i++;
	    if (r >= 4 && r < nc + 4) ins[r][r - 4] << i++;
	}
    }
    mtl::io::tout << name << ": A=\n" << A << '\n';
}

template <typename Matrix, typename Tag>
void two_d_iteration(const Matrix & A, Tag)
{
    namespace traits = mtl::traits;

    typename traits::row<Matrix>::type                                 row(A); 
    typename traits::col<Matrix>::type                                 col(A); 
    typename traits::const_value<Matrix>::type                         value(A); 
    typedef typename traits::range_generator<Tag, Matrix>::type        cursor_type;
    for (cursor_type cursor = mtl::begin<Tag>(A), cend = mtl::end<Tag>(A); cursor != cend; ++cursor) {
	typedef mtl::tag::nz     inner_tag;
	mtl::io::tout << "---\n";
	typedef typename traits::range_generator<inner_tag, cursor_type>::type icursor_type;
	for (icursor_type icursor = mtl::begin<inner_tag>(cursor), icend = mtl::end<inner_tag>(cursor); icursor != icend; ++icursor)
	    mtl::io::tout << "A[" << row(*icursor) << ", " << col(*icursor) << "] = " << value(*icursor) << '\n';
    }
} 


int main(int argc, char** argv)
{
    using namespace mtl;

    unsigned dim1= 3, dim2= 4;

    if (argc > 2) {dim1= atoi(argv[1]); dim2= atoi(argv[2]);}
    unsigned lsize= dim1 * dim2; 

    matrix::sparse_banded<double>  dr(lsize, lsize), dr2(6, 11), dr3(11, 6);
    
    rectangle_test(dr2, "Dense row major");
    rectangle_test(dr3, "Dense row major");
    laplacian_test(dr, dim1, dim2, "Dense row major");


    matrix::compressed2D<double> C;
    laplacian_setup(C, dim1, dim2);

    matrix::sparse_banded<double>  D;
    D= C;
    mtl::io::tout << "D is\n" << D;

    two_d_iteration(D, mtl::tag::row());

    matrix::compressed2D<double> E;
    E= D;
    std::cout << "D is\n" << D;




    return 0;
}
 
