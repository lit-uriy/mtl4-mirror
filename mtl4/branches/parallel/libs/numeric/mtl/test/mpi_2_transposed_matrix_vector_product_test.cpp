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


#include <boost/mpi.hpp>
#include <iostream>
#include <boost/serialization/string.hpp>
#include <boost/numeric/mtl/mtl.hpp>

//#include <boost/numeric/mtl/par/exception.hpp>

namespace mpi = boost::mpi;



template <typename Matrix, typename VectorIn, typename VectorOut>
void test(Matrix& A,  VectorIn& v, VectorOut& w, const char* name)
{
    mpi::communicator comm(communicator(A));

    // A= 0.0; // for dense matrices
    {
	mtl::matrix::inserter<Matrix> ins(A);
	mtl::vector::inserter<VectorIn> vins(v);
	if (comm.rank() == 0) {
	    ins[0][0] << 1.0;
	    ins[1][3] << 3.0;
	    ins[4][1] << 2.0; // remote
	    ins[6][5] << 4.0; // remote
	    vins[0] << 1.0;
	    vins[1] << 3.0;
	    vins[4] << 2.0; // remote
	    vins[6] << 4.0; // remote
	} else {
	    ins[2][6] << 5.0; // remote
	    ins[3][2] << 6.0; // remote
	    ins[5][4] << 8.0;
	    vins[2] << 5.0; // remote
	    vins[3] << 6.0; // remote
	    vins[5] << 8.0;
	}
    }

    mtl::par::single_ostream sout;
    sout << "Matrix is:\n" << A; sout.flush();
    sout << "v is: " << v << "\n";

    w= trans(A) * v;
    sout << "\nw= A * v is: " << w << '\n';
    if (std::abs(local(w)[1] - (comm.rank() ? 16.0 : 4.0)) > 0.01) throw "wrong value.";
}


int main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
   

    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
#if 0
	std::cerr << "Example for other than 2 processors!\n";
	mtl::par::mpi_error e(3);
	std::cerr << "e.what() = " << e.what() << '\n';
#endif
	return 0;
    } 


    std::vector<std::size_t> row_block, col_block;
    row_block.push_back(0); row_block.push_back(5); row_block.push_back(7); 
    col_block.push_back(0); col_block.push_back(4); col_block.push_back(7); 

    mtl::par::block_distribution row_dist(row_block), col_dist(col_block);

    matrix::distributed<matrix::compressed2D<double> > A(7, 7, row_dist, col_dist);
    vector::distributed<dense_vector<double> >         v(7, row_dist), w(7, col_dist);

    test(A, v, w, "compressed2D<double> * dense_vector<double>");
    
    return 0;
}

 














