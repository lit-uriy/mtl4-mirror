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

//#include <boost/numeric/mtl/operation/operators.hpp>
#include <boost/numeric/mtl/cuda/compressed2D.cu>
#include <boost/numeric/mtl/cuda/dense2D.cu>
#include <boost/numeric/mtl/cuda/dot.cu>
#include <boost/numeric/mtl/cuda/scalar.cu>
#include <boost/numeric/mtl/cuda/vector_cuda.cu>
#include <cmath>


namespace mtl { namespace cuda {

template < typename Vector>
void short_print2(const Vector& v)
{
   std::cout << "[";
   for (int i= 0; i < 10 && i < size(v); i++)
     std::cout << v[i] << ", ";
   if(size(v)> 20) {
       std::cout << " ... ";
       for (int i= size(v)-10; i < size(v); i++)
	   std::cout << v[i] << ", ";
   }
   std::cout << "\b\b] \n";
   v.to_device();
}
  
#define short_print(v) std::cout << #v << ' '; short_print2(v);
  
// Conjugate Gradients without preconditioner

template < typename LinearOperator, typename VectorX, typename VectorB >
int cg(LinearOperator& A, VectorX& x, VectorB& b, int iter, double tol)
{
  typedef typename mtl::Collection<VectorX>::value_type  Scalar;
  
  Scalar rho(0), rho_1(0), alpha(0), beta(0);
  VectorX p(size(x)), q(size(x)), r(size(x));  //z(size(x))
  
    x += p* alpha;
  
  #if 0
    r = b - A*x;
    rho= dot(r,r);
    int i(0);
    while ((sqrt(rho) > tol) && (i < iter)) {
	//       z = solve(M, r);
	if (i == 0)
	    p= r;
	else {
	    beta = rho / rho_1;
 	    p*= beta;
	    p= r+p;
	}
	A.mult(p, q);  // q = A * p;
	
	alpha = rho / dot(p, q);

 	x += p* alpha;
	r -= q* alpha;

	rho_1 = rho;      
	rho= (dot(r,r));
	++i;
	if (i%25 == 0) std::cout<< "iteration "<< i <<": norm residum=" << sqrt(rho) << "\n";
    }
    std::cout<< "\n\nAll done without problems\n";
   #endif	
    return iter;
}

}} // namespace mtl 

#endif // MTL_CUDA_CG_INCLUDE
