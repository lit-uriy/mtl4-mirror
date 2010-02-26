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

#if defined(MTL_HAS_MPI)

#include <map>
#include <utility>
#include <vector>
#include <algorithm>
#include <complex>

#include <boost/numeric/mtl/mtl.hpp>

#include <boost/mpi.hpp>
#include <boost/serialization/complex.hpp>

namespace mpi = boost::mpi;
typedef std::complex<double>           ct;

template <typename Matrix>
void test(Matrix& A,  const char* name)
{
    

    typedef typename mtl::Collection<Matrix>::size_type  size_type;
    typedef typename mtl::Collection<Matrix>::value_type value_type;
    typedef std::pair<size_type, size_type>              entry_type;
    typedef std::vector<entry_type>                      vec_type;

    mtl::par::single_ostream                             sout;
    mpi::communicator                                    comm(communicator(A));

    A= 2.0;
    sout << name << ": matrix is:\n" << A;

    typedef typename mtl::DistributedCollection<Matrix>::local_type local_type;
    local_type                                                      B(agglomerate(A));
    sout << "Agglomerated matrix is:\n" << B;

    if (comm.rank() == 0) {
	if (num_rows(A) != num_rows(B) || num_cols(A) != num_cols(B))
	    std::cerr << "wrong dimension in agglomeration!", throw "wrong dimension in agglomeration!";
	for (size_type i= 0; i < num_rows(B); i++)
	    for (size_type j= 0; j < num_cols(B); j++)
		if (B[i][j] != (i == j ? 2.0 : 0.0))
		    std::cerr << "wrong value!", throw "wrong value";
    }
}


int main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 2) {
	std::cerr << "Example works only for 2 processors!\n";
	env.abort(87);
    }
    std::vector<std::size_t> row_block, col_block;
    row_block.push_back(0); row_block.push_back(4); row_block.push_back(6); 
    col_block.push_back(0); col_block.push_back(5); col_block.push_back(7); 

    mtl::par::block_distribution row_dist(row_block), col_dist(col_block);

    matrix::distributed<matrix::compressed2D<double> > A(6, 7, row_dist, col_dist);
    matrix::distributed<matrix::compressed2D<ct> >     B(6, 7, row_dist, col_dist);
    matrix::distributed<matrix::dense2D<double> >      C(6, 7, row_dist, col_dist);

    test(A, "compressed2D<double>");
    test(B, "compressed2D<complex<double> >");
    test(C, "dense2D<double>");

    std::cout << "\n**** no errors detected\n";
    return 0;
}

 
#else 

int main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_MPI (and of course"
	      << " the presence of MPI).\n";
    return 0;
}

#endif












