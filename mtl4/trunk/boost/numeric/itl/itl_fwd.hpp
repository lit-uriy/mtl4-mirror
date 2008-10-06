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

#ifndef ITL_ITL_FWD_INCLUDE
#define ITL_ITL_FWD_INCLUDE

namespace itl {

    template <class Real> class basic_iteration;
    template <class Real> class noisy_iteration;

    template <typename Solver, typename VectorIn, bool trans> class solver_proxy;

    namespace pc {

	template <typename Matrix> class identity;
	template <typename Matrix> class diagonal;
	template <typename Matrix> class ilu_0;
	template <typename Matrix> class ic_0;

    } //  namespace pc

    template < typename LinearOperator, typename HilbertSpaceX, typename HilbertSpaceB, 
	       typename Preconditioner, typename Iteration >
    int cg(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
	   const Preconditioner& M, Iteration& iter);

    template < typename LinearOperator, typename Vector, 
	       typename Preconditioner, typename Iteration >
    int bicg(const LinearOperator &A, Vector &x, const Vector &b,
	     const Preconditioner &M, Iteration& iter);

    template < class LinearOperator, class HilbertSpaceX, class HilbertSpaceB, 
	       class Preconditioner, class Iteration >
    int bicgstab(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
		 const Preconditioner& M, Iteration& iter);

    template < class LinearOperator, class HilbertSpaceX, class HilbertSpaceB, 
	       class Preconditioner, class Iteration >
    int bicgstab_2(const LinearOperator& A, HilbertSpaceX& x, const HilbertSpaceB& b, 
		   const Preconditioner& M, Iteration& iter);

    template < typename LinearOperator, typename Vector, 
	       typename LeftPreconditioner, typename RightPreconditioner, 
	       typename Iteration >
    int bicgstab_ell(const LinearOperator &A, Vector &x, const Vector &b,
		     const LeftPreconditioner &L, const RightPreconditioner &R, 
		     Iteration& iter, size_t l);

} // namespace itl

#endif // ITL_ITL_FWD_INCLUDE
