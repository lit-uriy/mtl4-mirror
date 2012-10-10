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

#ifndef ITL_PC_CONCAT_INCLUDE
#define ITL_PC_CONCAT_INCLUDE

#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>
#include <boost/numeric/itl/pc/solver.hpp>

namespace itl { namespace pc {

/// Class for concatenating \tparam PC1 and \tparam PC2
template <typename PC1, typename PC2, typename Matrix, bool Store1= true, bool Store2= true>
class concat
{
    typedef typename boost::mpl::if_c<Store1, PC1, const PC2&>::type pc1_type;
    typedef typename boost::mpl::if_c<Store2, PC2, const PC2&>::type pc2_type;

  public:
    /// Construct both preconditioners from matrix \p A
    explicit concat(const Matrix& A) : pc1(A), pc2(A)
    {
	BOOST_STATIC_ASSERT((Store1 && Store2));
    }

    /// Both preconditioners are already constructed and passed as arguments
    /** If pc1 or pc2 is only constructed temporarily in the constructor call,
	the according Store argument must be true; otherwise the preconditioner
	will be a stale reference.
	Conversely, if the preconditioner is already build outside the constructor call,
	the according Store argument should be false for not storing the preconditioner twice. **/
    concat(const PC1& pc1, const PC2& pc2) : pc1(pc1), pc2(pc2) {}

  private:
    template <typename VectorOut>
    VectorOut& create_y0(const VectorOut& y) const
    {
	static VectorOut  y0(resource(y));
	return y0;
    }

  public:
    /// Solve P1 * P2 * x = y approximately by successive application (P2 first).
    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& x, VectorOut& y) const
    {
	mtl::vampir_trace<5058> tracer;
	y.checked_change_resource(x);
	VectorOut& y0= create_y0(y);

	pc2.solve(x, y0);
	pc1.solve(y0, y);
    }

    /// Solve adjoint(P1 * P2) * x = y approximately by successive application.
    /** Corresponds to adjoint(P2) * adjoint(P1) * x = y; thus adjoint(P1) is used applied first) **/
    template <typename VectorIn, typename VectorOut>
    void adjoint_solve(const VectorIn& x, VectorOut& y) const
    {
	mtl::vampir_trace<5059> tracer;
	y.checked_change_resource(x);
	VectorOut& y0= create_y0(y);

	pc1.adjoint_solve(x, y0);
	pc2.adjoint_solve(y0, y);
    }

   private:
    pc1_type   pc1;
    pc2_type   pc2;
};

template <typename PC1, typename PC2, typename Matrix, bool Store1, bool Store2, typename Vector>
solver<concat<PC1, PC2, Matrix, Store1, Store2>, Vector, false>
inline solve(const concat<PC1, PC2, Matrix, Store1, Store2>& P, const Vector& x)
{
    return solver<concat<PC1, PC2, Matrix, Store1, Store2>, Vector, false>(P, x);
}

template <typename PC1, typename PC2, typename Matrix, bool Store1, bool Store2, typename Vector>
solver<concat<PC1, PC2, Matrix, Store1, Store2>, Vector, true>
inline adjoint_solve(const concat<PC1, PC2, Matrix, Store1, Store2>& P, const Vector& x)
{
    return solver<concat<PC1, PC2, Matrix, Store1, Store2>, Vector, true>(P, x);
}


}} // namespace itl::pc

#endif // ITL_PC_CONCAT_INCLUDE
