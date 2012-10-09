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

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <typeinfo>

template <typename Matrix>
inline void strided_laplacian_setup(Matrix& A, unsigned m, unsigned n)
{
    A.change_dim(2*m*n, 2*m*n);
    set_to_zero(A);
    mtl::matrix::inserter<Matrix>      ins(A, 5);

    for (unsigned i= 0; i < m; i++)
	for (unsigned j= 0; j < n; j++) {
	    typename mtl::Collection<Matrix>::value_type four(4.0), minus_one(-1.0);
	    unsigned row= 2 * (i * n + j);
	    ins(row, row) << four;
	    if (j < n-1) ins(row, row+2) << minus_one;
	    if (i < m-1) ins(row, row+2*n) << minus_one;
	    if (j > 0) ins(row, row-2) << minus_one;
	    if (i > 0) ins(row, row-2*n) << minus_one;
	}
    for (unsigned i= 1; i < num_rows(A); i+= 2)
	ins(i, i) << 2;
}


int main()
{
    // For a more realistic example set sz to 1000 or larger
    const int size = 3, N = 2 * size * size; 

    typedef mtl::compressed2D<double>  matrix_type;
    typedef itl::pc::ic_0<matrix_type> ic_type;


    mtl::compressed2D<double>          A;
    strided_laplacian_setup(A, size, size);
    std::cout << "A is\n" << A << '\n';

#if 0
    itl::pc::sub_matrix_pc<ic_type, matrix_type> P(make_tag_vector(N, mtl::srange(0, iall, 2)), A);


  
    itl::pc::ic_0<matrix_type, float>  P(A);
    mtl::dense_vector<double>          x(N, 1.0), b(N);
    
    b = A * x;
    x= 0;

    itl::cyclic_iteration<double> iter(b, N, 1.e-6, 0.0, 1);
    cg(A, x, b, P, iter);
    
    // test(mtl::lazy(b)= solve(P, x));
#endif 
    return 0;
}
