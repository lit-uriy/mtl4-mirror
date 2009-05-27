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

#ifndef MTL_COMMUNICATOR_INCLUDE
#define MTL_COMMUNICATOR_INCLUDE

#ifdef MTL_HAS_MPI

#include <boost/mpi.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl {

namespace vector {

    /// Communicator of assign expression
    /** Returns communicator of first argument, i.e. assigned vector **/
    template <typename E1, typename E2, typename SFunctor> 
    inline boost::mpi::communicator const& communicator(const vec_vec_aop_expr<E1, E2, SFunctor>& expr)  
    {
	return communicator(expr.first);
    }

    /// Communicator of binary plus/minus expression
    /** Returns communicator of first argument **/
    template <typename E1, typename E2, typename SFunctor> 
    inline boost::mpi::communicator const& communicator(const vec_vec_pmop_expr<E1, E2, SFunctor>& expr)
    {
	return communicator(expr.first.value);
    }

    /// Communicator of mapped view
    template <typename Functor, typename Coll>
    inline boost::mpi::communicator const& communicator(const map_view<Functor, Coll>& expr)
    {
	return communicator(expr.ref);
    }

} // vector

} // namespace mtl

#endif

#endif // MTL_COMMUNICATOR_INCLUDE
