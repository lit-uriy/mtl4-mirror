// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University.
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG (haftungsbeschränkt), www.simunova.com.
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#include <iostream>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>
#include <boost/numeric/itl/krylov/async_executor.hpp>

#include <chrono>
#include <thread>

int main()
{
    mtl::vampir_trace<9999> tracer;
    // For a more realistic example set size to 1000 or larger
    const int size = 300, N = size * size;

    typedef mtl::compressed2D<double>  matrix_type;
    mtl::compressed2D<double>          A(N, N);
    laplacian_setup(A, size, size);

    itl::pc::identity<matrix_type>     P(A);

    mtl::dense_vector<double> x(N, 1.0), b(N);

    b = A * x;
    x= 0;
    const int iter_limit= 100000;
    itl::cyclic_iteration<double> iter(b, iter_limit, 0.0, 0.0, iter_limit);
    
    auto my_cg_solver= itl::make_cg_solver(A, P);
    itl::async_executor<decltype(my_cg_solver)> async_exec(my_cg_solver);
    
    // auto async_ex= itl::make_async_executor(itl::make_cg_solver(A, P));
    async_exec.start_solve(x, b, iter);
    
    std::this_thread::sleep_for( std::chrono::milliseconds( 50 ) ); 
    async_exec.interrupt();
    
    if (iter.iterations() == iter_limit)
        throw "Executor was not interrupted!\n";
    else
        std::cout << "Yeah, it was interrupted!!!\n";

    return 0;
}
