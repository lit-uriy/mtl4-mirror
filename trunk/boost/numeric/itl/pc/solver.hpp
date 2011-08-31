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

#ifndef ITL_PC_SOLVER_INCLUDE
#define ITL_PC_SOLVER_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/numeric/mtl/vector/assigner.hpp>

namespace itl { namespace pc {

template <typename PC, typename Vector, bool adjoint= false>
struct solver
  : mtl::vector::assigner<solver<PC, Vector> >
{
    typedef PC  pc_type;
    solver(const pc_type& P, const Vector& x) : P(P), x(x) {}

    template <typename VectorOut>
    void assign_to(VectorOut& y) const
    {	assign_to(y, boost::mpl::bool_<adjoint>());    }    

  protected:

    template <typename VectorOut>
    void assign_to(VectorOut& y, boost::mpl::false_) const
    {	P.solve(x, y);    }    
    
    template <typename VectorOut>
    void assign_to(VectorOut& y, boost::mpl::true_) const
    {	P.adjoint_solve(x, y);    }    

    const pc_type&        P; 
    const Vector&         x;
};

}} // namespace itl::pc

#endif // ITL_PC_SOLVER_INCLUDE
