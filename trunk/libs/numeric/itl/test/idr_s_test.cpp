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

#include <cmath>

#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>


struct Random
{
  std::complex<double> operator()() const
  {
    return std::complex<double>(
        (2.*(static_cast<double>(rand())/RAND_MAX - 0.5)),
        (2.*(static_cast<double>(rand())/RAND_MAX - 0.5))
        );
  }
};

template <typename Matrix>
void test1(Matrix& m, double tau)
{
  Random m_rand;
  mtl::matrix::inserter<Matrix> ins(m);
  size_t nrows=num_rows(m);
  std::complex<double> val;
  for (size_t r=0;r<nrows;++r)
  {
    for (size_t c=0;c<nrows;++c)
    {
      if(r==c) 
        ins(r,c) << 1.;
      else
      {
        val=m_rand();
        if (abs(val)<tau)
          ins(r,c) << val;  
      } 
    }
  } 
}

int test_main(int argc, char* argv[])
{
  const int N = 12; // Original from Jan had 2000 
  const int Niter = 3*N;

  typedef mtl::compressed2D<double> matrix_type;
  //typedef compressed2D<std::complex<double> ,matrix::parameters<tag::col_major> > matrix_type;
  matrix_type                   A(N, N);
  laplacian_setup(A,3,4);
  mtl::dense_vector<double> b(N), x(N, 1.0);
  b=A*x;

  itl::pc::identity<matrix_type>     Ident(A);
   
   x= 5.0;
  itl::cyclic_iteration<double> iter_1(b, Niter, 1.e-8, 0.0, 1);
  idr_s(A, x, b, Ident, Ident, iter_1,4);
#if 0
  std::cout << "Non-preconditioned bicgstab(2)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_2b(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, Ident, iter_2b,2);

  std::cout << "Non-preconditioned bicgstab(4)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_4b(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, Ident, iter_4b,4);

  std::cout << "Non-preconditioned bicgstab(8)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_8b(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, Ident, iter_8b,8);

  itl::pc::ilu_0<matrix_type>        P(A);
  
  std::cout << "Right ilu(0) preconditioned bicgstab(1)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_1r(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, P, iter_1r,1);
 
  std::cout << "Right ilu(0) preconditioned bicgstab(2)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_2r(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, P, iter_2r,2);

  std::cout << "Left ilu(0) preconditioned bicgstab(4)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_4l(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, P, Ident, iter_4l,4);

  std::cout << "Right ilu(0) preconditioned bicgstab(4)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_4r(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, P, iter_4r,4);

  std::cout << "Right ilu(0) preconditioned bicgstab(8)" << std::endl;
  x= 0.5;
  itl::cyclic_iteration<double> iter_8r(b, Niter, 1.e-8, 0.0, 5);
  idr_s(A, x, b, Ident, P, iter_8r,8);
#endif
  return 0;
}
