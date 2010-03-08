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
#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/scalar.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>


int main()
{
  // For a more realistic example set size to 1000 or larger
  const int size = 4, N = size * size;

  typedef mtl::cuda::dense2D<double>  matrix_type;
  matrix_type         A(N, N);
  
  A.set_to_zero();
  
  //Laplacian Setup
  for(int i=  1; i < N; i++) {
      A(4, i, i);
      A(-1,i-1,i);
      A(-1, i, i-1);
  }
  A(4,0,0); A(4, N-1, N-1);
    
  mtl::cuda::vector<double> x(N, 1), b(N, 0);
  
  std::cout<< "A=\n" << A << "\n";

  b = A * x;
  std::cout<< "b=" << b << "\n";
  x= 0;

  double toleranz=   0.000001;
  int	 iterations= 1000;
  
  cg(A, x, b, iterations, toleranz);

  return 0;
}
