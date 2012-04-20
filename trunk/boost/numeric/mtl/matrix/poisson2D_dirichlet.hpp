// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University.
//               2010 SimuNova UG, www.simunova.com. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also tools/license/license.mtl.txt in the distribution.

#ifndef MTL_MATRIX_POISSON2D_DIRICHLET_INCLUDE
#define MTL_MATRIX_POISSON2D_DIRICHLET_INCLUDE

#include <boost/numeric/mtl/utility/ashape.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/matrix/multiplier.hpp>

namespace mtl { namespace matrix {

struct poisson2D_dirichlet
{
    poisson2D_dirichlet(int m, int n) : m(m), n(n) {}

    template <typename VectorIn, typename VectorOut, typename Assign>
    void mult(const VectorIn& v, VectorOut& w, Assign) const
    {
	MTL_DEBUG_THROW_IF(int(size(v)) == m * n, incompatible_size());
	MTL_DEBUG_THROW_IF(size(v) == size(w), incompatible_size());

	// Inner domain
	for (int i= 1; i < m-1; i++)
	    for (int j= 1, k= i * n + j; j < n-1; j++, k++) 
		Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k-1] - v[k+1]); 
	    
	// Upper border
	for (int j= 1; j < n-1; j++) 
	    Assign::apply(w[j], 4 * v[j] - v[j+n] - v[j-1] - v[j+1]);

	// Lower border
	for (int j= 1, k= (m-1) * n + j; j < n-1; j++, k++) 
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k-1] - v[k+1]); 
	
	// Left border
	for (int i= 1, k= n; i < m-1; i++, k+= n)
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k+1]); 

	// Right border
	for (int i= 1, k= n+n-1; i < m-1; i++, k+= n)
	    Assign::apply(w[k], 4 * v[k] - v[k-n] - v[k+n] - v[k-1]); 

	// Corners
	Assign::apply(w[0], 4 * v[0] - v[1] - v[n]);
	Assign::apply(w[n-1], 4 * v[n-1] - v[n-2] - v[2*n - 1]);
	Assign::apply(w[(m-1)*n], 4 * v[(m-1)*n] - v[(m-2)*n] - v[(m-1)*n+1]);
	Assign::apply(w[m*n-1], 4 * v[m*n-1] - v[m*n-2] - v[m*n-n-1]);
    }

    template <typename VectorIn>
    multiplier<poisson2D_dirichlet, VectorIn> operator*(const VectorIn& v) const
    {	return multiplier<poisson2D_dirichlet, VectorIn>(*this, v);    }

    int m, n;
};


}} // namespace mtl::matrix

namespace mtl { namespace ashape {
    template <> struct ashape_aux<matrix::poisson2D_dirichlet> 
    {	typedef nonscal type;    };
}}

#endif // MTL_MATRIX_POISSON2D_DIRICHLET_INCLUDE
