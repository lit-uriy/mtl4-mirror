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

#ifndef MTL_VEC_ACOS_INCLUDE
#define MTL_VEC_ACOS_INCLUDE

#include <boost/numeric/mtl/vector/map_view.hpp>

namespace mtl { namespace vec {

    /// Element-wise aco of \a v
    template <typename Vector>
    acos_view<Vector> acos(const Vector& v)
    {
        return acos_view<Vector>(v);
    }



}} // namespace mtl::vec

#endif // MTL_VEC_ACOS_INCLUDE
