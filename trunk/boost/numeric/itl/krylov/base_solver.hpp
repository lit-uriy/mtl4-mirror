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

    base_solver(const LinearOperator& A, iteration_type iter= default_iteration<magnitude_type>()) 
      : A(A), my_iteration(iter) 
    { my_iteration.set_quite(true); }

    /// Perform one iteration on linear system
    template <typename HilbertSpaceB, typename HilbertSpaceX>
    int step(HilbertSpaceX& x, const HilbertSpaceB& b) const
    {
	itl::basic_iteration<double> iter(b, 1, 0, 0);
	return static_cast<Solver const*>(this)->solve(x, b, iter);
    }
	
    /// Solve using the iteration (resets initial residuum)
    template <typename HilbertSpaceB, typename HilbertSpaceX>
    int operator()(HilbertSpaceX& x, const HilbertSpaceB& b) 
    {
	if (two_norm(x) == 0)
	    my_iteration.set_norm_r0(two_norm(b));
	else {
	    HilbertSpaceB r(b);
	    r-= A * x;
	    my_iteration.set_norm_r0(two_norm(r));
	}
	return static_cast<Solver const*>(this)->solve(x, b, my_iteration);
    }

    /// Return iteration object
    iteration_type iteration() const { return my_iteration; }

    /// Reference to iteration object (to be used with care)
    iteration_type& iteration_ref() { return my_iteration; }

  protected:
    const LinearOperator& A;
    iteration_type my_iteration;
};


} // namespace itl

#endif // ITL_BASE_SOLVER_INCLUDE
