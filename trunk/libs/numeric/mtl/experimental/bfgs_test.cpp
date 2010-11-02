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

#include <iostream>
#include <utility>
#include <cmath>
#include <boost/test/minimal.hpp>
#include <boost/numeric/mtl/mtl.hpp>
#include <boost/numeric/itl/itl.hpp>

using namespace std;  
   
template <typename Vector>
Vector grad_f(Vector& x)
{    
   Vector tmp(size(x));
   tmp[0]=2*x[0];
   tmp[1]=4*x[1];
   tmp[2]=-4*x[2];
   return tmp;
}


template <typename value_type>
value_type f(mtl::dense_vector<value_type>& x)
{
  return x[0]*x[0] + 2*x[1]*x[1] - 2*x[2]*x[2];
}

template <typename value_type>
value_type armijo(mtl::dense_vector<value_type>& x, mtl::dense_vector<value_type>& d)
{
  value_type delta= 0.5, gamma= 0.5, beta1= 0.25, beta2= 0.75;  //feste Werte
  //Star_Schrittweite
  value_type alpha= -gamma*dot(grad_f(x), d)/dot(d, d);
  mtl::dense_vector<value_type> tmp(size(x));
  tmp= x+alpha*d;
  while ( f(tmp) > (f(x) + delta*alpha*dot(grad_f(x), d)) ) 
  {
    alpha= (beta1*alpha+beta2*alpha)/2;
    tmp= x+alpha*d;
  }
  return alpha;
}

template <typename Matrix, typename Vector>
Matrix bfgs(Matrix& H_old, Vector& y, Vector& s)
{
  double gamma= 1/dot(y,s);
  int r= num_rows(H_old), c= num_cols(H_old);
  Matrix H(r,c), I(r,c), I2(r,c);
  H= 0; I= 1; I2= 1; 
  y*=-1*gamma;
  rank_one_update(I, s, y); rank_one_update(I2, y, s);
  rank_one_update(H, s, s);
  H*= gamma;
  H+= I * H_old * I2;
  return H;
}
 

template <typename Vector>
Vector quasi_newton(Vector& x, double tol)
{    
   Vector d_k, y_k, x_k, s_k;
   mtl::dense2D<double>  H(size(x),size(x));
   H= 1;   //H0 ist Einheitsmatrix
   
   while (two_norm(grad_f(x)) > tol) {
   //for(int i = 0; i < 5; i++){
    // std::cout<< "grad_f = " << two_norm(grad_f(x)) << "\n";
     d_k= -1*H*grad_f(x);
     std::cout<< "d_k = " << d_k << "\n";
     double alpha= armijo(x, d_k);
     std::cout<< "alpha = " << alpha << "\n";
     x_k= x + alpha * d_k;
     s_k= alpha*d_k;
     y_k= grad_f(x_k) - grad_f(x);
     H= bfgs(H, y_k, s_k);
     std::cout<< "H = \n" << H << "\n";
     x= x_k;
   }
   return x;
}

int test_main(int, char**)
{
    using namespace mtl;

    mtl::dense_vector<double>       x0(3, 8);
    double tol= 1e-4;
    std::cout<< "x0= " << x0 << "\n";
    
    quasi_newton(x0, tol);
    std::cout<< "x0= " << x0 << "\n";
    std::cout<< "grad_f(x0)= " << grad_f(x0) << "\n";
    

    return 0;
}
 














