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
    

    dense2D<double>    A(io::matrix_market(fname.c_str()));
    //    cout << "Size of A is " << num_rows(A) << " x " << num_cols(A) << '\n';
   
    dense2D<double>    C(hessenberg_factors(A)), D(clone(bands(C, -1, 2)));
    cout << "The tridiagonal matrix is\n" << D[irange(10)][irange(10)];
   

    

    return 0;
}
