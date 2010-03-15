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

#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

#include <boost/test/minimal.hpp>
#include <boost/mpi.hpp>

namespace mpi = boost::mpi;

#define MTL_RUN_SOLVER( name, solver, argp, args)			\
    {									\
	sos << name << "\n";						\
	x= 0.01;							\
	int codep= solver argp;						\
	xs= 0.01;							\
	int codes= solver args;						\
        if (codes != 0)							\
	    sos << "Sequential code doesn't even converge!!!\n";	\
	else if (codep != 0)						\
	    sos << "Parallel code doesn't converge!!\n";		\
	else if (iter.iterations() >= 2 * iters.iterations())		\
	    sos << "Parallel code converges too slowly!\n";		\
    }
    


template <typename Stream, typename Vector>
void start(Stream& s, Vector& x, const char* name)
{
    s << name << "\n";
    x= 0.0;
}

int test_main(int argc, char* argv[])
{
    // For a more realistic example set size to 1000 or larger
    const int size = 4, N = size * size;
    
    mpi::environment env(argc, argv);

    typedef mtl::matrix::distributed<mtl::compressed2D<double> > matrix_type;
    matrix_type          A(N, N);
    laplacian_setup(A, size, size);
    
    itl::pc::ic_0<matrix_type>      L(A);
    itl::pc::identity<matrix_type>  R(A);
    
    mtl::vector::distributed<mtl::dense_vector<double> > x(N, 1.0), b(N); 
    
    b= A * x;
    
    mtl::par::single_ostream sos;
    itl::cyclic_iteration<double, mtl::par::single_ostream> iter(b, N/2, 1.e-6, 0.0, 5, sos);
    const unsigned ell= 6, restart= ell, s= ell;

    sos << "A is\n" << agglomerate(A) << "two_norm(b) is " << two_norm(b) << '\n';

    typedef mtl::compressed2D<double>  matrix_s_type;
    matrix_s_type                                           As(N, N);
    laplacian_setup(As, size, size);
    mtl::dense_vector<double>                               xs(N, 1.0), bs(N);
    bs= As * xs;
    itl::pc::ic_0<matrix_s_type>                            Ls(As);
    itl::pc::identity<matrix_s_type>                        Rs(As);
    itl::cyclic_iteration<double, mtl::par::single_ostream> iters(bs, N/2, 1.e-6, 0.0, 5, sos);

    MTL_RUN_SOLVER("Bi-Conjugate Gradient", bicg, (A, x, b, L, iter), (As, xs, bs, Ls, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized", bicgstab, (A, x, b, L, iter), (As, xs, bs, Ls, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized(2)", bicgstab_2, (A, x, b, L, iter), (As, xs, bs, Ls, iters));
    MTL_RUN_SOLVER("Bi-Conjugate Gradient Stabilized(ell)", bicgstab_ell, (A, x, b, L, R, iter, ell), (As, xs, bs, Ls, Rs, iters, ell));
    MTL_RUN_SOLVER("Conjugate Gradient", cg, (A, x, b, L, iter), (As, xs, bs, Ls, iters));
    MTL_RUN_SOLVER("Conjugate Gradient Squared", cgs, (A, x, b, L, iter), (As, xs, bs, Ls, iters));
    // MTL_RUN_SOLVER("Generalized Minimal Residual method (without restart)", gmres_full, (A, x, b, L, R, iter, size), (As, xs, bs, Ls, iters, size));
    // MTL_RUN_SOLVER("Generalized Minimal Residual method with restart", gmres, (A, x, b, L, R, iter, restart), (As, xs, bs, Ls, iters, restart));
    // MTL_RUN_SOLVER("Induced Dimension Reduction on s dimensions (IDR(s))", idr_s, (A, x, b, L, R, iter, s), (As, xs, bs, Ls, iters, s));
    MTL_RUN_SOLVER("Quasi-minimal residual", qmr, (A, x, b, L, R, iter), (As, xs, bs, Ls, Rs, iters));
    MTL_RUN_SOLVER("Transposed-free Quasi-minimal residual", tfqmr, (A, x, b, L, R, iter), (As, xs, bs, Ls, Rs, iters));

#if 0
    start(sos, x, "Bi-Conjugate Gradient");                                 bicg(A, x, b, L, iter);
    start(sos, x, "Bi-Conjugate Gradient Stabilized");                      bicgstab(A, x, b, L, iter);
    start(sos, x, "Bi-Conjugate Gradient Stabilized(2)");                   bicgstab_2(A, x, b, L, iter);
    start(sos, x, "Bi-Conjugate Gradient Stabilized(ell)");                 bicgstab_ell(A, x, b, L, R, iter, ell); // refactor with multi_vector?
    start(sos, x, "Conjugate Gradient");                                    cg(A, x, b, L, iter);
    start(sos, x, "Conjugate Gradient Squared");                            cgs(A, x, b, L, iter); 
    // start(sos, x, "Generalized Minimal Residual method (without restart)"); gmres_full(A, x, b, L, R, iter, size);
    // start(sos, x, "Generalized Minimal Residual method with restart");      gmres(A, x, b, L, R, iter, restart);
    // start(sos, x, "Induced Dimension Reduction on s dimensions (IDR(s))");  idr_s(A, x, b, L, R, iter, s); 
    start(sos, x, "Quasi-minimal residual");                                qmr(A, x, b, L, R, iter);  
    start(sos, x, "Transposed-free Quasi-minimal residual");                tfqmr(A, x, b, L, R, iter);
#endif

    return 0;
}
