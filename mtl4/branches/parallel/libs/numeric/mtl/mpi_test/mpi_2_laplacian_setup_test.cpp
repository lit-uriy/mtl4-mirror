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
#include <boost/test/minimal.hpp>
#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;

using namespace std;  

template <typename Matrix>
void test(Matrix& matrix, unsigned dim1, unsigned dim2, const char* name)
{
    mtl::par::single_ostream sout;
    sout << "\n" << name << "\n";
    laplacian_setup(matrix, dim1, dim2);
    sout << "Laplacian matrix:\n" << matrix << "\n";
    
    mpi::communicator comm(communicator(matrix));

    if (comm.rank() == 0 && dim1 == 3 && dim2 == 4) {
	typename mtl::Collection<Matrix>::value_type four(4.0), minus_one(-1.0), zero(0.0);
	if (local(matrix)[0][0] != four)
	    throw "wrong diagonal";
	if (local(matrix)[0][1] != minus_one)
	    throw "wrong east neighbor";
	if (local(matrix)[0][dim2] != minus_one)
	    throw "wrong south neighbor";
	if (dim2 > 2 && local(matrix)[0][2] != zero)
	    throw "wrong zero-element";
    }
}



int test_main(int argc, char* argv[])
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }

    unsigned dim1= 3, dim2= 4;

    if (argc > 2) {
	dim1= atoi(argv[1]); 
	dim2= atoi(argv[2]);
    }
    unsigned size= dim1 * dim2; 

    matrix::distributed<dense2D<double> >                                      dr(size, size);
    matrix::distributed<dense2D<double, matrix::parameters<col_major> > >      dc(size, size);
    matrix::distributed<morton_dense<double, recursion::morton_z_mask> >       mzd(size, size);
    matrix::distributed<morton_dense<double, recursion::doppled_2_row_mask> >  d2r(size, size);
    matrix::distributed<compressed2D<double> >                                 cr(size, size);
    matrix::distributed<compressed2D<double, matrix::parameters<col_major> > > cc(size, size);

    test(dr, dim1, dim2, "Dense row major");
    test(dc, dim1, dim2, "Dense column major");
    test(mzd, dim1, dim2, "Morton Z-order");
    test(d2r, dim1, dim2, "Hybrid 2 row-major");
    test(cr, dim1, dim2, "Compressed row major");
    test(cc, dim1, dim2, "Compressed column major");

    return 0;
}
