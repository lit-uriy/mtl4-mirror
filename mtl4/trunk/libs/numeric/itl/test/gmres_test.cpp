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

// Written by Cornelius Steinhardt

#include <cmath>

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>




template <typename Matrix>
void test1(Matrix& m, double tau)
{
  mtl::matrix::inserter<Matrix> ins(m);
  size_t nrows=num_rows(m);
  double val;
  for (size_t r=0;r<nrows;++r)
  {
    for (size_t c=0;c<nrows;++c)
    {
      if(r==c)
        ins(r,c) << 1.;
      else
      {
        val=2.*(static_cast<double>(rand())/RAND_MAX - 0.5);
        if (val<tau)
          ins(r,c) << val;
      }
    }
  }
}


int test_main(int argc, char* argv[])
{

  const int N = 10; // Original from Jan had 2000
  const int Niter = 3*N;

  typedef mtl::dense2D<double> matrix_type;
  //typedef compressed2D<std::complex<double>matrix::parameters<tag::col_major> > matrix_type;
  matrix_type                   A(N, N);
  //laplacian_setup(A, N, N);
  mtl::dense_vector<double> b(N, 1), x(N);

  test1(A,0.194);
  std::cout << "A has " << A.nnz() << " non-zero entries" << std::endl;
  itl::pc::identity<matrix_type>     Ident(A);

  std::cout << "Non- preconditioned bicgstab" << std::endl;
  std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_0(b, Niter, 1.e-8);

  bicgstab(A, x, b, Ident, iter_0);
  std::cout << "x= " << x << " \n\n" ;
  //std::cout << "\n Non-preconditioned gmres(1)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_1(b, Niter, 1.e-8);
  gmres(A, x, b,iter_1,1);
  std::cout << "x= " << x << " \n" ;

  //std::cout << "\n Non-preconditioned gmres(2)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_2(b, Niter, 1.e-8);
  gmres(A, x, b,iter_2,2);

  //std::cout << "\n Non-preconditioned gmres(4)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_4(b, Niter, 1.e-8);
  gmres(A, x, b,iter_4,4);

  //std::cout << "\n Non-preconditioned gmres(8)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_8(b, Niter, 1.e-8);
  gmres(A, x, b,iter_1,8);


  //std::cout << "\n Non-preconditioned gmres(16)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_16(b, Niter, 1.e-8);
  gmres(A, x, b,iter_16,16);

  //std::cout << "\n Non-preconditioned gmres(32)" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_32(b, Niter, 1.e-8);
  gmres(A, x, b,iter_32,32);

//
  return 0;
}
