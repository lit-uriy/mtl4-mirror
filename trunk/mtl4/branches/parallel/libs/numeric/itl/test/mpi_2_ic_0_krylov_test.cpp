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

// #undef MTL_ASSERT_FOR_THROW // Don't wont to abort program when one solver fail

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

#include <boost/test/minimal.hpp>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

#define MTL_RUN_SOLVER( name, solver, argp, args)			\
    {									\
	sos << "\n\n" << name << "\n";					\
	x= 0.01;							\
	itl::cyclic_iteration<double, mtl::par::single_ostream> iter(b, N, 1.e-4, 0.0, 5, sos); \
	int codep, codes;						\
        try {								\
	    codep= solver argp;						\
	} catch (...) {							\
	    std::cerr << "Error in parallel solver!\n";			\
	    codep= 10;							\
	}								\
									\
	xs= 0.01;							\
        itl::cyclic_iteration<double, mtl::par::single_ostream> iters(bs, N, 1.e-4, 0.0, 10, sos); \
        try {								\
	    codes= solver args;						\
	} catch (...) {							\
	    std::cerr << "Error in sequential  solver!\n";		\
	    codes= 10;							\
	}								\
									\
        if (codes != 0)							\
	    sos << "Sequential code doesn't even converge!!!\n", failed++; \
	else if (codep != 0)						\
	    sos << "Parallel code doesn't converge!!\n", failed++;	\
	else if (iter.iterations() >= 2 * iters.iterations())		\
	    sos << "Parallel code converges too slowly!\n", failed++;	\
	else								\
	    succeed++;							\
    }
    

int test_main(int argc, char* argv[])
{
    // For a more realistic example set size to 1000 or larger
    const int size = 4, N = size * size;
    int       succeed= 0, failed= 0;
    
    mpi::environment env(argc, argv);

    typedef mtl::matrix::distributed<mtl::compressed2D<double> > matrix_type;
    matrix_type          A(N, N);
    laplacian_setup(A, size, size);

    itl::pc::ilu_0<matrix_type>     ILU(A);
    itl::pc::ic_0<matrix_type>      IC(A);
    itl::pc::identity<matrix_type>  I(A);
    
    mtl::vector::distributed<mtl::dense_vector<double> > x(N, 1.0), b(N); 
    
    b= A * x;
    
    mtl::par::single_ostream sos;
    const unsigned ell= 6, restart= ell, s= ell;

    sos << "A is\n" << agglomerate(A) << "two_norm(b) is " << two_norm(b) << '\n';

    typedef mtl::compressed2D<double>  matrix_s_type;
    matrix_s_type                                           As(N, N);
    laplacian_setup(As, size, size);
    mtl::dense_vector<double>                               xs(N, 1.0), bs(N);
    bs= As * xs;
    itl::pc::ilu_0<matrix_s_type>                           ILUs(As);
    itl::pc::ic_0<matrix_s_type>                            ICs(As);
    itl::pc::identity<matrix_s_type>                        Is(As);

#if 0
    MTL_RUN_SOLVER("Bi-Conjugate Gradient", bicg, (A, x, b, I, iter), (As, xs, bs, Is, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized", bicgstab, (A, x, b, ILU, iter), (As, xs, bs, ILUs, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized(2)", bicgstab_2, (A, x, b, ILU, iter), (As, xs, bs, ILUs, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized(ell)", bicgstab_ell, (A, x, b, ILU, I, iter, ell), (As, xs, bs, ILUs, Is, iters, ell));
    MTL_RUN_SOLVER("Conjugate Gradient", cg, (A, x, b, IC, iter), (As, xs, bs, ICs, iters));
    MTL_RUN_SOLVER("Conjugate Gradient Squared", cgs, (A, x, b, ILU, iter), (As, xs, bs, ILUs, iters));
#endif
    MTL_RUN_SOLVER("Generalized Minimal Residual method (without restart)", gmres_full, (A, x, b, I, I, iter, size), (As, xs, bs, Is, Is, iters, size));
    MTL_RUN_SOLVER("Generalized Minimal Residual method with restart", gmres, (A, x, b, I, I, iter, restart), (As, xs, bs, Is, Is, iters, restart));
    // MTL_RUN_SOLVER("Induced Dimension Reduction on s dimensions (IDR(s))", idr_s, (A, x, b, ILU, I, iter, s), (As, xs, bs, ILUs, Is, iters, s));
#if 0
    MTL_RUN_SOLVER("Quasi-minimal residual", qmr, (A, x, b, ILU, I, iter), (As, xs, bs, ILUs, Is, iters));
    MTL_RUN_SOLVER("Transposed-free Quasi-minimal residual", tfqmr, (A, x, b, ILU, I, iter), (As, xs, bs, ILUs, Is, iters));
#endif
    sos << succeed << " solvers succeeded and " << failed << " solvers failed.\n";

    return 0;
}
