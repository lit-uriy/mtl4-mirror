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
  std::cout<< "CG START\n"; 
  typedef typename mtl::Collection<VectorX>::value_type  Scalar;
  scalar<Scalar> rho(0), rho_1(0), alpha(0), beta(0), temp(0);
  VectorX p(size(x)), q(size(x)), r(size(x)), z(size(x)), s(size(x)), t(size(x));

    double norm(1);
    p= A*x;
    r= b - p;	
    norm= sqrt(dot(r,r));
    int i(0);
    while ((norm > tol) && (i < iter)) {
	//       z = solve(M, r);
	rho = dot(r, r);
	if (i == 0)
	    p= r;
	else {
	    beta = rho.value() / rho_1.value();
	    p*= beta.value();
	    p= r+p;
	}	
	q = A * p;
	// temp = dot(p, q);
	alpha = rho.value() / dot(p, q);
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
	norm= sqrt(dot(r,r));
	if (i % 20 == 0)
	    std::cout<< "iteration "<< i <<": norm residum=" << norm << "\n";
    }
    std::cout<< "\n\nAll without problems\n";
    return iter;
}

}} // namespace mtl 

#endif // MTL_CUDA_CG_INCLUDE
