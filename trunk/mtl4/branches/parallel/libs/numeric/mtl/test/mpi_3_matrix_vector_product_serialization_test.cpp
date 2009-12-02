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

// Works for arbitrary processor numbers, even larger then number of matrix rows

#include <iostream>
#include <cstdlib>

#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>

namespace mpi = boost::mpi;

template <typename Matrix, typename Vector>
void test(Matrix& A, Vector& v, Vector& w, const char* name)
{
    mpi::communicator comm(communicator(A));

    unsigned n= num_rows(A);
    // A= 0.0; // for dense matrices
    {
	mtl::matrix::inserter<Matrix> ins(A);
	mtl::vector::inserter<Vector> vins(v);
	if (comm.rank() == 0) {
	    for (unsigned i= 0; i < n; i++) {
		vins[i] << rand();
		for (unsigned j= n-2; j <= n; j++) 
		    ins[i][(i+j) % n] << rand();
	    }
	}
    }
    mtl::par::single_ostream sout;
    if (n < 10)
	sout << "Matrix is:\n" << A << "\nv is: " << v << "\n";

    w= A * v;
    if (n < 10)
	sout << "Parallel computation of A * v is: " << w << "\n";

    // Agglomerated values
    mtl::par::block_migration    migr= agglomerated_migration(row_distribution(A));
    Vector va(v, migr), wa(w, migr);
    Matrix Aa(A, migr);

    // Local comparison on first processor
    if (comm.rank() == 0) {
	typedef typename mtl::DistributedCollection<Vector>::local_type local_type;
	local_type ws(n);  
	ws= local(Aa) * local(va);
	if (n < 10)
	    std::cout << "Sequential computation of A * v is: " << ws << "\n";
	
	using std::abs;
	if (one_norm(local_type(local(wa) - ws)) / one_norm(ws) > 0.00001) { // threshold to high for rand?
	    std::cerr << "Parallel and sequential computation different!\n";
	    throw "Parallel and sequential computation different!\n";
	}
    }
}


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);
    mpi::communicator world;

    int size= 7;
    if (argc > 1) size= atoi(argv[1]);
    if (size <= 2) {
	std::cerr << "Example works only with matrices of size 3x3 or greater!\n";
	env.abort(87);
    }

    matrix::distributed<matrix::compressed2D<double> > A(size, size);
    vector::distributed<dense_vector<double> >         v(size), w(size);

    test(A, v, w, "compressed2D<double> * dense_vector<double>");
    
    return 0;
}

 














