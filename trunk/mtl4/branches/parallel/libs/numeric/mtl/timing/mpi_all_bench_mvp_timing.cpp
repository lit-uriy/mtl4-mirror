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
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/timer.hpp>

namespace mpi = boost::mpi;


int test_main(int argc, char* argv[]) 
{
    using namespace mtl;

    mpi::environment env(argc, argv);

    std::string program_dir= mtl::io::directory_name(argv[0]);
    std::string fname(argc > 1 ? argv[1] : "matrix_market/laplace_3x4.mtx");

    typedef matrix::distributed<matrix::compressed2D<double> >   matrix_type;

    mtl::par::timer read_time;
    //     matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market(fname));
    // TEST Matrix
    matrix::distributed<matrix::compressed2D<double> > A(mtl::io::matrix_market("/home/simunova/csteinhardt/mtl4-parallel/libs/numeric/mtl/test/matrix_market/laplace_3x4.mtx"));
    //BENCHMARK Matrix
    // matrix_type A(mtl::io::matrix_market("/home/simunova/csteinhardt/mtl4-parallel/libs/numeric/mtl/test/matrix_market/Ga41As41H72.mtx"));

    mtl::par::sout << "read_time is  " << read_time.elapsed() << "sec (not micros)\n";
    mtl::vector::distributed<mtl::dense_vector<double> > u, v(num_cols(A), 3.0);

#ifdef MTL_HAS_PARMETIS
    mtl::par::timer migt;
    mtl::par::block_migration migrator= parmetis_migration(A);
    matrix_type B(A, migrator);
    mtl::vector::distributed<mtl::dense_vector<double> > u2, v2(v, migrator);
    
    mtl::par::sout << "time for migration is " << migt.elapsed()*1000000.0 << " micros \n";
#endif
    
    mtl::par::timer mvp_time;
    u= A*v;
    mtl::par::sout << "time for MVP is " << mvp_time.elapsed()*1000000.0 << " micros \n";
    // mtl::par::sout << "u is  " << u << '\n';

#ifdef MTL_HAS_PARMETIS
    mtl::par::timer mvpr_time;
    u2= B*v2;
    mtl::par::sout << "time for MVP repartitioned is " << mvpr_time.elapsed()*1000000.0 << " micros \n";
#endif

    mtl::par::sout << "timer resolution is " << mvp_time.elapsed_min() << '\n';
    

    return 0;
}

 














