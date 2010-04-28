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

using namespace std;
int main()
{
  // For a more realistic example set size to 1000 or larger
  const int size = 10, N = size * size;

//   typedef mtl::dense2D<double>  matrix_type;
  typedef mtl::compressed2D<double>  matrix_type;
  matrix_type                        A(5,5);

  //laplacian_setup(A, size, size);

  A= 1,  1,  0,   0,  0,
      1, -1, -2,   0,  0,
      0, -2,  1,   7,  0,
      0,  0,  5, -10,  9,
      0,  0,  0,   3, 22;

  itl::pc::identity<matrix_type>     P(A);
  mtl::dense_vector<double>          x(5, 1.0), b(5), test(5);
  mtl::dense_vector<complex<double> > xz(5,complex<double>(1.0, 0.0)), bz(5);

  b= A * x;
  x= 0;
  
  itl::cyclic_iteration<double> iter(b, 500, 1.e-6, 0.0, 1);
  thomas(A, x, b, iter);
  test= b-A*x;
  std::cout<< "norm rest=" << two_norm(test) << "\n\n";
  
  x=0;
  itl::cyclic_iteration<double> iterb(b, 500, 1.e-6, 0.0, 1);
  bicgstab(A, x, b, P, iterb);
//   std::cout<< "x=" << x << "\n";
  test= b-A*x;
  std::cout<< "norm rest=" << two_norm(test) << "\n";
  
  return 0;
}


