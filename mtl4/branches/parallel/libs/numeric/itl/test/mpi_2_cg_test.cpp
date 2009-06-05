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

int test_main(int argc, char* argv[])
{
  // For a more realistic example set size to 1000 or larger
  const int size = 10, N = size * size;

  typedef mtl::matrix::distributed<mtl::compressed2D<double> > matrix_type;
  matrix_type          A(N, N);
  laplacian_setup(A, size, size);

  itl::pc::diagonal<matrix_type>     P(A);
  //itl::pc::identity<matrix_type>     P(A);

  mtl::vector::distributed<mtl::dense_vector<double> > x(N, 1.0), b(N);

  b = A * x;
  x= 0;

  mtl::par::single_ostream sos;
  itl::cyclic_iteration<double, mtl::par::single_ostream> iter(b, 500, 1.e-6, 0.0, 5, sos);
  cg(A, x, b, P, iter);

  return 0;
}
