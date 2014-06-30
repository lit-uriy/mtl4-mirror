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

#ifndef ITL_BASE_SOLVER_INCLUDE
#define ITL_BASE_SOLVER_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/concept/magnitude.hpp>
#include <boost/numeric/itl/iteration/cyclic_iteration.hpp>
#include <boost/numeric/itl/iteration/default_iteration.hpp>

namespace itl {

template <typename Solver, typename LinearOperator>
struct base_solver
{
    typedef typename mtl::Magnitude<typename mtl::Collection<LinearOperator>::value_type>::type magnitude_type;
    typedef cyclic_iteration<magnitude_type>                                                    iteration_type;

    base_solver(iteration_type i= default_iteration<magnitude_type>()) : my_iteration(i) { my_iteration.set_quite(true); }

    /// Perform one iteration on linear system
    template <typename HilbertSpaceB, typename HilbertSpaceX>
    int step(HilbertSpaceX& x, const HilbertSpaceB& b) const
    {
	itl::basic_iteration<double> iter(b, 1, 0, 0);
	return static_cast<Solver const*>(this)->solve(x, b, iter);
    }
	
    /// Solve using default iteration
    template <typename HilbertSpaceB, typename HilbertSpaceX>
    int operator()(HilbertSpaceX& x, const HilbertSpaceB& b) const
    {
	iteration_type iter= my_iteration;
	iter.set_norm_r0(two_norm(b));
	return static_cast<Solver const*>(this)->solve(x, b, iter);
    }

    iteration_type my_iteration;
};


} // namespace itl

#endif // ITL_BASE_SOLVER_INCLUDE
