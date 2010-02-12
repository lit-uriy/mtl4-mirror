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
void test(Matrix& A,  const char* name, int version)
{
    typedef typename mtl::Collection<Matrix>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    mpi::communicator comm(communicator(A));
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins, 10*(comm.rank()+1));
        switch (version) {
          case 1: 
	    switch (comm.rank()) {
	      case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); std::cout << "version 1\n"; break;
    	      case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); break;
    	      case 2: i(5, 6); i(6, 4); i(6, 5);
    	    }; break;
          case 2: 
    	    switch (comm.rank()) {
	      case 0: i(0, 1); i(1, 2); i(2, 3); std::cout << "\n\nversion 2\n"; break;
    	      case 1: i(3, 4); i(4, 5); break;
    	      case 2: i(5, 6); i(6, 0);
    	  }; break;
        }
    }

    sout << "Matrix is:" << '\n' << A;

    mtl::par::block_migration migration= parmetis_migration(A);
    Matrix B(num_rows(A), num_cols(A), migration.new_distribution());
    migrate_matrix(A, B, migration);

    sout << "Migrated matrix is:\n" << B;

    Matrix C(A, parmetis_migration(A)); 
    sout << "Migrated matrix (in constructor) is:\n" << C;
}


int main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);

    
#if 0
    std::string    program_dir= mtl::io::directory_name(argv[0]), fname(argc > 1 ? argv[1] : "matrix_market/mhd1280b.mtx");
    matrix::distributed<matrix::compressed2D<double> >             C(mtl::io::matrix_market(mtl::io::join(program_dir, fname)));
#endif     

    matrix::distributed<matrix::compressed2D<double> >  A(7, 7);

    test(A, "distributed<compressed>", 1); 
    test(A, "distributed<compressed>", 2); 

    mtl::par::single_ostream sout;

    std::cout << "\n**** no errors detected\n";
    return 0;
}
 
#else 

int main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_PARMETIS (and of course"
	      << " the presence of ParMetis).\n";
    return 0;
}

#endif












