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
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <boost/timer.hpp>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

int main(int argc, char* argv[])
{
    typedef mtl::matrix::distributed<mtl::compressed2D<double> > matrix_type;

    mpi::environment env(argc, argv);
    mpi::communicator world;

    const bool strong= true;
    int size1= strong ? 1000 : 1000 * world.size(), size2= 1000, N= size1 * size2;

    boost::timer time;
    matrix_type          A(N, N);
    laplacian_setup(A, size1, size2);
    double assembly_time= time.elapsed();
    
    time.restart();
    itl::pc::ic_0<matrix_type>     P(A);
    double pc_time= time.elapsed();
    
    mtl::vector::distributed<mtl::dense_vector<double> > x(N, 1.0), b(N); 
    
    b= A * x;
    x= 0;
    
    mtl::par::single_ostream sos;
    itl::cyclic_iteration<double, mtl::par::single_ostream> iter(b, 500, 1.e-6, 0.0, 50, sos);

    time.restart();
    cg(A, x, b, P, iter);
    double solve_time= time.elapsed();

    sos << (strong ? "Strong" : "Weak") << " scaling on " << world.size() << " processors"
	<< ", Assembly: " << assembly_time << "s, Preconditioner setup: " << pc_time
	<< "s, Solver: " << solve_time << "s\n";
    sos << world.size() << ", " << assembly_time << ", " << pc_time
	<< ", " << solve_time << "\n";
    
    return 0;
}
