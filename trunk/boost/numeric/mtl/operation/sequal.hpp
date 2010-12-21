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

#ifndef MTL_VECTOR_SEQUAL_INCLUDE
#define MTL_VECTOR_SEQUAL_INCLUDE

#include <cmath>
#include <boost/utility.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/operation/diagonal.hpp>
#include <boost/numeric/mtl/operation/givens.hpp>
#include <boost/numeric/mtl/operation/qr.hpp>
#include <boost/numeric/mtl/operation/rank_one_update.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>

#include <boost/numeric/mtl/vector/dense_vector.hpp>


namespace mtl { namespace vector {

/// 
template <typename Vector>
class sequal
{
    typedef typename Collection<Vector>::value_type   value_type;
    typedef typename Collection<Vector>::size_type    size_type;

  public:
    /// 
    sequal(Vector& lambda, Vector& z, Vector& d, value_type sigma) : lambda(lambda), z(z), d(d), sigma(sigma), fw(1), gfw(0)
    {
	using std::abs;
	value_type zero= math::zero(sigma);
	lambda= zero; 
	std::cout<<"konstruktor lambda  ="<< lambda <<"\n";
	}

    /// 
    value_type& funk(const value_type& lamb)
    {
	    fw=1;
	    for(size_type i=0; i<size(z); i++){
	      fw+=sigma*z[i]*z[i]/(d[i]-lamb);
	    }

	    return fw;
    }

     /// 
    value_type& grad_f(const value_type& lamb)
    {
	    gfw= 0.0;
	    for(size_type i=0; i<size(z); i++){
			gfw+=sigma*(z[i]/(d[i]-lamb))*(z[i]/(d[i]-lamb));
	    }

	    return gfw;
    }
    
     /// 
    Vector& roots(const value_type& lamb)
    {
		//need sorted d
	   std::cout<<"konstruktor lambda  ="<< lambda <<"\n";
	   Vector start(size(z), 0.0);
	   for(size_type i=0; i<size(z)-1; i++){
			start[i]= (d[i]+d[i+1])/2;  //startpunkte innerhalb 2er pole
	   }
	   start[size(z)-1]= d[size(z)-1] + 5.0;  // start punkt rechts vom letzten pol
	   std::cout<< "start with=" << start << "\n";
	    for(size_type i=0; i<size(z); i++){
	      lambda[i]= newton(start[i]); //newton algo
	    }

	    return lambda;
    }

	/// 
    value_type newton(value_type start)
    {
		double tol= 1.0e-5;
		value_type lamb(start);
	    while (std::abs(funk(lamb)) > tol){
			lamb-= funk(lamb)/grad_f(lamb);	
			std::cout<<"lamb loop ="<< lamb <<"\n";
			std::cout<<"f(lamb) ="<< funk(lamb) <<"\n";
	    }

	    return lamb;
    }



  private:
    Vector&    lambda, z, d;
    value_type sigma, fw, gfw;
};

}}// namespace vector


#endif // MTL_VECTOR_SEQUAL_INCLUDE

