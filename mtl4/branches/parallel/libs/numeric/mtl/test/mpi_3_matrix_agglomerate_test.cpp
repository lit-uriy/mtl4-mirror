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
void test(Matrix& A,  const char* name)
{
    typedef typename mtl::Collection<Matrix>::size_type size_type;
    typedef std::pair<size_type, size_type>             entry_type;
    typedef std::vector<entry_type>                     vec_type;

    mtl::par::single_ostream sout;
    mtl::par::multiple_ostream<> mout;

    std::vector<idxtype> part;
    mpi::communicator comm(communicator(A));
    A= 0;
    {
	mtl::matrix::inserter<Matrix> mins(A);
	ins<mtl::matrix::inserter<Matrix> > i(mins, 10*(comm.rank()+1));
	switch (comm.rank()) {
	  case 0: i(0, 1); i(0, 2); i(1, 2); i(1, 3); i(2, 3); i(2, 5); 
	      part.push_back(1); part.push_back(0); part.push_back(1); break;
	  case 1: i(3, 4); i(3, 5); i(4, 5); i(4, 6); 
	      part.push_back(0); part.push_back(0); break;
	  case 2: i(5, 6); i(6, 4); i(6, 5);
	      part.push_back(2); part.push_back(2);
	}; 
    }

    sout << "Matrix is:\n" << A;
    sout << "Agglomerated matrix is:\n" << agglomerate(A);

    typedef typename mtl::DistributedCollection<Matrix>::local_type local_type;
    local_type B(agglomerate(A));

    if (comm.rank() == 0) {
	if (B[0][0] != 0) std::cerr << "B[0][0] != 0\n", throw 1;
	if (B[0][2] != 11) std::cerr << "B[0][2] != 11\n", throw 1;
	if (B[0][3] != 0) std::cerr << "B[0][3] != 0\n", throw 1;
	if (B[3][0] != 0) std::cerr << "B[3][0] != 0\n", throw 1;
	if (B[3][4] != 20) std::cerr << "B[3][4] != 20\n", throw 1;
	if (B[3][5] != 21) std::cerr << "B[3][5] != 21\n", throw 1;
	if (B[3][6] != 0) std::cerr << "B[3][6] != 0\n", throw 1;
	if (B[5][6] != 30) std::cerr << "B[5][6] != 30\n", throw 1;
    }
}


int main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    if (world.size() != 3) {
	std::cerr << "Example works only for 3 processors!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(7, 7);
    matrix::distributed<matrix::dense2D<double> >      B(7, 7);

    test(A, "compressed2D<double>");
    test(B, "dense2D<double>");

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












