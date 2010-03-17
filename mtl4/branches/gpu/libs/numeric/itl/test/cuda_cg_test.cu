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
#include <boost/numeric/mtl/cuda/cg.cu>
#include <boost/numeric/mtl/cuda/config.cu>
#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/compressed2D.cu>
#include <boost/numeric/mtl/cuda/scalar.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>


int main(int argc, char* argv[])
{
  using namespace mtl;

  // For a more realistic example set size to 1000 or larger
  const int size = 100, N = size * size;
  mtl::cuda::activate_best_gpu();
  typedef mtl::cuda::compressed2D<double>  matrix_type;
  
  matrix_type         A(N, N);
  //std::cout<< "Start Matrix("<< N <<"x"<< N<<") set_to_zero\n";
  A.set_to_zero();
 // std::cout<< "A=" << A << "\n";
  std::cout<< "Ende Matrix set_to_zero\n Start init Laplacian setup";

  //Laplacian Setup
  A.simpel_laplacian_setup(size+2*(size-1), 4);
  mtl::cuda::vector<double> x(N, 1), b(N, 2);
  x.to_device();
  
//    std::cout<< "A=\n" << A << "\n";

b = A * x;
//    std::cout<< "b=" << b << "\n";
//   std::cout<< "x=" << x << "\n";
  x= 0;

  double toleranz=   0.0000001;
  int	 iterations= 1000;
  
  cg(A, x, b, iterations, toleranz);
  

  return 0;
}
