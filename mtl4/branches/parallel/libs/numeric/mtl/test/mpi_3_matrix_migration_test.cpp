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

#if defined(MTL_HAS_PARMETIS) && defined(MTL_HAS_MPI)

#include <map>
#include <utility>
#include <vector>
#include <algorithm>


#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector and std::pair
#include <boost/numeric/mtl/mtl.hpp>

#include <parmetis.h>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>

namespace mpi = boost::mpi;

template <typename Inserter>
struct ins
{
    typedef typename Inserter::size_type  size_type;
    ins(Inserter& i, int start) : i(i), v(start) {}
    void operator()(size_type r, size_type c) {	i[r][c] << double(v++);  }
    Inserter& i;
    int       v;
};

template <typename Matrix>
inline void cv(const Matrix& A, unsigned r, unsigned c, double v)
{
    //if (A[r][c] != v) throw "Wrong value;";
}

template <typename Matrix>
void test(Matrix& A,  const char* name, int version)
{
    typedef typename mtl::Collection<Matrix>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    std::vector<idxtype> part;
    mpi::communicator comm(communicator(A));
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins, 10*(comm.rank()+1));
        switch (version) {
          case 1: 
	    switch (comm.rank()) {
	      case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); std::cout << "version 1\n"; 
		      part.push_back(1); part.push_back(0); part.push_back(1); break;
    	      case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); 
                      part.push_back(0); part.push_back(0); break;
    	      case 2: i(5, 6); i(6, 4); i(6, 5);
		      part.push_back(2); part.push_back(2);
    	    }; break;
          case 2: 
    	    switch (comm.rank()) {
	      case 0: i(0, 1); i(1, 2); i(2, 3); std::cout << "\n\nversion 2\n"; 
		      part.push_back(0); part.push_back(1); part.push_back(1); break;
    	      case 1: i(3, 4); i(4, 5); 
                      part.push_back(1); part.push_back(2); break;
    	      case 2: i(5, 6); i(6, 0);
		      part.push_back(2); part.push_back(0);
    	  }; break;
        }
    }

    sout << "Matrix is:\n" << A;

    mtl::par::block_migration    migration= parmetis_migration(row_distribution(A), part);
    Matrix B(7, 7, migration.new_distribution());
    migrate_matrix(A, B, migration);

    sout << "Migrated matrix is:\n" << B;

    switch (version) {
      case 1: 
	switch (comm.rank()) {
          case 0: cv(B, 0, 1, 13.); cv(B, 0, 4, 12.); cv(B, 1, 2, 20.); cv(B, 1, 5, 21.);
	          cv(B, 2, 5, 22.); cv(B, 2, 6, 23.); break;
	  case 1: cv(B, 3, 0, 10.); cv(B, 3, 4, 11.); cv(B, 4, 1, 14.); cv(B, 4, 5, 15.); break;
	  case 2: cv(B, 5, 6, 30.); cv(B, 6, 2, 31.); cv(B, 6, 5, 32.); 
	}; break;
     case 2: 
       switch (comm.rank()) {
	 case 0: cv(B, 0, 2, 10.); cv(B, 1, 0, 31.); break;
	 case 1: cv(B, 2, 3, 11.); cv(B, 3, 4, 12.); cv(B, 4, 5, 20.); break;
	 case 2: cv(B, 5, 6, 21.); cv(B, 6, 1, 30.);
       }; break;
    }

    mtl::par::block_migration    rev= reverse(migration);
    Matrix C(B, rev);

    sout << "Back-migrated matrix is (should be the original one (not automatically tested yet)):\n" << C;

}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(7, 7), B(7, 7);

    test(A, "compressed2D<double>", 1);
    test(B, "compressed2D<double>", 2);

    return 0;
}

 
#else 

int test_main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
    return 0;
}

#endif












