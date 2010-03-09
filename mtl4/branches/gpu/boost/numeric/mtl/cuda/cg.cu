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

#ifndef MTL_CUDA_CG_INCLUDE
#define MTL_CUDA_CG_INCLUDE

#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/dot.cu>
#include <boost/numeric/mtl/cuda/scalar.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <cmath>


namespace mtl { namespace cuda {

/// Conjugate Gradients without preconditioner
template < typename LinearOperator, typename VectorX, typename VectorB >
int cg(LinearOperator& A, VectorX& x, const VectorB& b, int iter, double tol)
{
  std::cout<< "CG START\n"; 
  typedef typename mtl::Collection<VectorX>::value_type  Scalar;
  scalar<Scalar> rho(0), rho_1(0), alpha(0), beta(0), temp(0);
  VectorX p(size(x)), q(size(x)), r(size(x)), z(size(x)), s(size(x)), t(size(x));
  double norm(1);
  
   r.to_device(); x.to_device(); b.to_device();
  //A.to_device();
 // std::cout<< "A=" << A << "\n";
  
  r = A*x;
  r= r - b;
//   std::cout<< "r="<< r << "\n"; 
  norm= sqrt(dot(r,r));
  int i(0);
  std::cout<< "dot(r,r)="<< dot(r,r) << "\n";
  std::cout<< "norm="<< norm << "\n";
  while ((norm > tol) && (i < iter)) {
//       z = solve(M, r);
      rho = dot(r, r);
//      	  std::cout<< "i="<< i << "\n";
      if (i == 0){
	  p = r;
      } else {
//  	  std::cout<< "rho="<< rho << "\n";
//  	  std::cout<< "rho_1="<< rho_1 << "\n";
	  beta = rho.value();
//  	  std::cout<< "beta="<< beta << "\n";
	  beta/= rho_1.value();
//  	  std::cout<< "beta="<< beta << "\n";
// 	  p = z + beta * p;
	  p*= beta.value();
// 	  std::cout<< "p="<< p << "\n";
	  p= r+p;
      }
//         std::cout<< "2.i="<< i << "\n";
      
      q = A * p;
//       std::cout<< "A="<< A << "\n";
//       std::cout<< "p="<< p << "\n";
//       std::cout<< "q="<< q << "\n";
      temp = dot(p, q);
      alpha = rho;
      alpha/= temp.value();
      
      s= p;
      t= q;
      
      s*=  alpha.value();
      t*=  alpha.value();
      //x += alpha * p;
      //r -= alpha * q;
      x= x + s;
      r= r - t;
      rho_1 = rho;
      
      ++i;
     // std::cout<< "dot(r,r)="<< dot(r,r) << "\n";
      norm= sqrt(dot(r,r));
      std::cout<< "iteration "<< i <<": norm residum=" << norm << "\n";
  }
  return iter;
}

}} // namespace mtl 

#endif // MTL_CUDA_CG_INCLUDE
