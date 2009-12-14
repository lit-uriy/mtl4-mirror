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

#if defined(MTL_HAS_PARMETIS) && defined(MTL_HAS_MPI) // && defined(MTL_HAS_TOPOMAP)

#define MTL_HAS_STD_OUTPUT_OPERATOR // to print std::vector
#include <boost/numeric/mtl/operation/std_output_operator.hpp>

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <boost/mpi.hpp>
#include <boost/timer.hpp>
#include <string>

using namespace std;
namespace mpi= boost::mpi;

template <typename Matrix>
void solve(const Matrix& A, const char* name)
{
    // Output on rank 0 only
    typedef mtl::par::single_ostream so_type;
    so_type sout;
    sout << name << '\n';

    // Distribution and size of A
    size_t n= num_rows(A);
    mtl::par::block_distribution dist(row_distribution(A));

    // Create vectors
    mtl::vector::distributed<mtl::dense_vector<double> >  x(n, dist), b(n, dist);

    // Set up linear system such that solution is x == 1.0
    x= 1.0;
    b= A * x;
    x= 0.0;

    if (n < 10)
	sout << "Matrix is:\n" << A << "\nb is: " << b << "\n";

    // Incomplete Cholesky for A
    itl::pc::ic_0<Matrix>     P(A);
    
    // Solve with CG
    itl::cyclic_iteration<double, so_type> iter(b, 50 /* max iter */, 1.e-8 /* rel. error red. */, 
						0.0, 30 /* how often logged */, sout);
    boost::timer time;
    cg(A, x, b, P, iter);
    sout << "Solution took " << time.elapsed() << "s.\n";
    if (n < 50)
	sout << "Solution is:\n" << x << "\nShould be Vector of 1s\n";
}



int test_main(int argc, char* argv[]) 
{
    using namespace mtl; using namespace mtl::par;

    typedef matrix::distributed<compressed2D<double> > matrix_type;

    mpi::environment env(argc, argv);
    mpi::communicator world;
    
    mtl::par::single_ostream     sout;
    mtl::par::multiple_ostream<> mout;

    // Set file name (consider program being started from other directory)
    std::string program_dir= io::directory_name(argv[0]), 
	file_name= io::join(program_dir, "matrix_market/bcsstk01.mtx"); // symm. (hopefully pos. def.)

    // Read file into distributed matrix
    io::matrix_market file(file_name);
    matrix_type A(file);
    assert(num_rows(A) == num_cols(A)); // check symmetry
    
    //matrix_type C(io::matrix_market("matrix.mtx")); // The file is not there!!!
    solve(A, "Matrix with naive block distribution");

    // Get partition from Parmetis (explicitly without topomap)
    parmetis_index_vector part;
    parmetis_partition_k_way(A, part);
    mout << "Metis partition is " << part << '\n'; 

    mtl::par::block_migration pmigr= parmetis_migration(row_distribution(A), part);
    mout << "New distribution is " << pmigr.new_distribution() << '\n';
    
    // Migrate matrix as parmetis says
    matrix_type B(A, parmetis_migration(row_distribution(A), part));
    solve(B, "Matrix migrated by Parmetis");
#if 0
    // Migrate as Torsten says
    topology_mapping(part);
    matrix_type C(A, parmetis_migration(row_distribution(A), part));
    solve(C, "Matrix migrated by Parmetis and topology mapping");
#endif    
    // It could have been so easy if we wouldn't compare the two mappings ;-) 
    // matrix_type D(C, parmetis_migration(C));

    return 0;
}

 
#else 

int test_main(int argc, char* argv[]) 
{
    std::cout << "Test requires the definition of MTL_HAS_TOPOMAP and MTL_HAS_PARMETIS(and of course"
	      << " the presence of Topology Mapping and ParMetis).\n";
    return 0;
}

#endif












