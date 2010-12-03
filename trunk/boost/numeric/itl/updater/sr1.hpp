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

#ifndef ITL_SR1_INCLUDE
#define ITL_SR1_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>

namespace itl {

/// Update of Hessian matrix for e.g. Quasi-Newton by SR1 formula
struct sr1
{
    /// 
    template <typename Matrix, typename Vector>
    void operator() (Matrix& H, const Vector& y, const Vector& s)
    {
	assert(num_rows(H) == num_cols(H));
	Vector     d(s - H * y);
	H+= 1 / dot(d, y) * d * trans(d);
   }
}; 



} // namespace itl

#endif // ITL_SR1_INCLUDE
