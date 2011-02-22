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

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>


using namespace std;

template <typename Matrix>
void inline givens_elimination(Matrix& A)
{
    mtl::matrix::givens<Matrix> g(A, 1, 3);
    
}


int main(int argc, char** argv) 
{
    using namespace mtl;

    int select= 1, sub= 6;
    if (argc > 1)
	select= atoi(argv[1]);
    assert(select >= 1 && select <= 2);

    if (argc > 2)
	sub= atoi(argv[2]);
    
    string fname= string("../../../../../branches/data/matrix_market/Partha") + char('0' + select) + ".mtx";
    

    dense2D<double>    A0(io::matrix_market(fname.c_str())), A(clone(A0[irange(sub)][irange(sub)])), B(A[irange(sub)][irange(sub)]);
    // cout << "A[irange(sub)][irange(sub)] is\n" << B;

    cout << "Size of A is " << num_rows(A) << " x " << num_cols(A) << '\n'
	 << "A\n" << A << '\n';
   

    dense2D<double>    C(hessenberg(A)), D(extract_hessenberg(C));
    cout << "Hessenberg is\n" << C[irange(10)][irange(10)]
	 << "A is now\n" << A << '\n';
   

    dense2D<double>    E(hessenberg(B));// , F(extract_hessenberg(E));
    cout << "Hessenberg is\n" << E[irange(10)][irange(10)]
	 << "A is now\n" << A << '\n';



    return 0;
}
