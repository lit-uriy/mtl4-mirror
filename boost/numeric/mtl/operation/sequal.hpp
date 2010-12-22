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

// With contributions from Cornelius Steinhardt

#ifndef MTL_VECTOR_SECULAR_INCLUDE
#define MTL_VECTOR_SECULAR_INCLUDE

#include <cmath>
#include <boost/utility.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>


namespace mtl { namespace vector {

/// Class for the secular equation( to solve eigenvalue problems)
template <typename Vector>
class secular_f
{
    typedef typename Collection<Vector>::value_type   value_type;
    typedef typename Collection<Vector>::size_type    size_type;

  public:
    /// Construktor needs 3 Vectors lambda(to save roots), z(nummerator), d(dominator) and sigma as factor befor the sum
    secular_f(const Vector& lambda, const Vector& z, const Vector& d, value_type sigma) 
      : lambda(lambda), z(z), d(d), sigma(sigma)
    {
	this->lambda= math::zero(sigma); 
    }

    /// secular_f equation as function, evaluates the function value
    /** \f$f(x)=1+\sigma * sum_{i=1}^{n}\frac{z_i}{d_i-x} \f$**/
    value_type funk(const value_type& lamb)
    {
	value_type fw= 1;
	for(size_type i=0; i<size(z); i++)
	    fw+= sigma*z[i]*z[i]/(d[i]-lamb);
	return fw;
    }

    /// gradient of secular_f equation as function, evaluates the gradientfunction value
    /** \f$gradf(x)=\sigma * sum_{i=1}^{n}\frac{z_i}{(d_i-x)^2} \f$**/
    value_type grad_f(const value_type& lamb)
    {
	value_type gfw= 0.0;
	for(size_type i=0; i<size(z); i++)
	    gfw+= sigma*(z[i]/(d[i]-lamb))*(z[i]/(d[i]-lamb));  //TODO
	return gfw;
    }
    
    /// evaluates the roots of secular_f equation =0 with newton algo
    Vector roots()
    {
	assert(size(z) > 1);
	//need sorted d
	//Construct start values for newton iteration
	Vector start(size(z), 0.0);
	for(size_type i=0; i<size(z)-1; i++)
	    start[i]= (d[i] + d[i+1]) / 2;  //start points between pols
	start[size(z)-1]= 1.5 * d[size(z)-1] - 0.5 * d[size(z)-2];  // last start point plus half the distance to second-last

	//newton algo to find roots
	for(size_type i=0; i<size(z); i++)
	    lambda[i]= newton(start[i]); 
	return lambda;
    }

    /// newton algorithm x_{n+1}= x_n - f(x_n)/f'(x_n)
    value_type newton(value_type start)
    {
	double tol= 1.0e-5;  // TODO tol evtl parameter
	value_type lamb(start);
	while (std::abs(funk(lamb)) > tol)
	    lamb-= funk(lamb)/grad_f(lamb);	
	return lamb;
    }

  private:
    Vector    lambda, z, d;
    value_type sigma;
};

template <typename Vector, typename Value>
inline Vector secular(const Vector& lambda, const Vector& z, const Vector& d, Value sigma)
{
    secular_f<Vector> functor(lambda, z, d, sigma);
    return functor.roots();
}

}}// namespace vector


#endif // MTL_VECTOR_SECULAR_INCLUDE

