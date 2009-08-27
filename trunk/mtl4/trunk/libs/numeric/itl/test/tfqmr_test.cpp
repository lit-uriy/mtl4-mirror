
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
#include <boost/numeric/mtl/operation/trans.hpp>




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

  const int N = 200; // Original from Jan had 2000
  const int Niter = 5*N;

using itl::pc::identity;using itl::pc::ilu_0;
  typedef mtl::dense2D<double> matrix_type;
  matrix_type                   A(N, N);
  mtl::dense_vector<double> b(N, 1), x(N);
  identity<matrix_type>     Ident(A);


  test1(A,0.194);
  std::cout << "A has " << A.nnz() << " non-zero entries" << std::endl;


  std::cout << "Non-preconditioned tfqmr" << std::endl;
  std::cout << "Won't convergence (for large examples)!" << std::endl;
  x= 0.5;
  itl::noisy_iteration<double> iter_2(b, Niter, 1.e-8);
  tfqmr(A, x, b,iter_2);
  //std::cout << "x=" <<x<<"\n";

  //ilu_0<matrix_type>        P(A);
  //std::cout << "Non-preconditioned qmr" << std::endl;
  //std::cout << "Won't convergence (for large examples)!" << std::endl;
  //x= 0.5;
  //itl::noisy_iteration<double> iter_3(b, Niter, 1.e-8);
  //qmr(A, x, b,Ident,Ident,iter_3);


  return 0;
}



