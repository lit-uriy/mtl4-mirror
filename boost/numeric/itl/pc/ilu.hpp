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

#ifndef ITL_PC_ILU_INCLUDE
#define ITL_PC_ILU_INCLUDE

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/operation/lower_trisolve.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>
#include <boost/numeric/mtl/operation/lu.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/dense2D.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>


namespace itl { namespace pc {

template <typename Matrix, typename Factorizer, typename Value= typename mtl::Collection<Matrix>::value_type>
class ilu
{
  public:
    typedef Value                                         value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef ilu                                           self;
    typedef Factorizer                                    factorizer_type;

    typedef mtl::matrix::parameters<mtl::row_major, mtl::index::c_index, mtl::non_fixed::dimensions, false, size_type> para;
    typedef mtl::compressed2D<value_type, para>                     L_type;
    typedef mtl::compressed2D<value_type, para>                     U_type;

    typedef mtl::matrix::detail::lower_trisolve_t<L_type, mtl::tag::unit_diagonal, true>    lower_solver_t;
    typedef mtl::matrix::detail::upper_trisolve_t<U_type, mtl::tag::inverse_diagonal, true> upper_solver_t;

    // Factorization adapted from Saad
    ilu(const Matrix& A) : f(A, L, U), lower_solver(L), upper_solver(U) {}

    // Solve  LU y = x --> y= U^{-1} L^{-1} x
    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	Vector y;
	solve(x, y);
	return y;
    }

    // solve x = L y --> y0= L^{-1} x
    template <typename VectorIn, typename VectorOut>
    const VectorOut& solve_lower(const VectorIn& x, VectorOut&) const
    {
	static VectorOut y0(resource(x));
	lower_solver(x, y0);
	return y0;
    }

    // Solve  LU y = x --> y= U^{-1} L^{-1} x
    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& x, VectorOut& y) const
    {
	mtl::vampir_trace<5039> tracer;
	const VectorOut& y0= solve_lower(x, y);

	y.checked_change_resource(x);
	upper_solver(y0, y);
    }


    // Solve (LU)^H x = b --> x= L^{-H} U^{-H} b
    template <typename Vector>
    Vector adjoint_solve(const Vector& b) const
    {
	mtl::vampir_trace<5040> tracer;
	return unit_upper_trisolve(adjoint(L), inverse_lower_trisolve(adjoint(U), b));
    }


    L_type get_L() { return L; }
    U_type get_U() { return U; }

  public:
    L_type                      L;
    U_type                      U;
  private:
    Factorizer                  f;
    lower_solver_t              lower_solver;
    upper_solver_t              upper_solver;
}; 

template <typename Value, typename Factorizer, typename V2>
class ilu<mtl::dense2D<Value>, Factorizer, V2> // last 2 arguments are dummies
{
  public:
    typedef mtl::dense2D<Value>                           Matrix;
    typedef typename mtl::Collection<Matrix>::value_type  value_type;
    typedef typename mtl::Collection<Matrix>::size_type   size_type;
    typedef ilu                                           self;
    typedef Matrix                                        LU_type;

    ilu(const Matrix& A) : LU(A) { lu(LU, P); std::cout << "LU is\n" << LU << "P is " << P << "\n"; }

    // Solve  P^{-1}LU x = b --> x= U^{-1} L^{-1} P b
    template <typename Vector>
    Vector solve(const Vector& b) const { return lu_apply(LU, P, b); }

    // Solve  P^{-1}LU x = b --> x= U^{-1} L^{-1} P b
    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& b, VectorOut& x) const { x= lu_apply(LU, P, b); }

    // Solve (P^{-1}LU)^H x = b --> x= P^{-1}L^{-H} U^{-H} b // P^{-1}^{-1}^H = P^{-1})
    template <typename Vector>
    Vector adjoint_solve(const Vector& b) const { return lu_adjoint_apply(LU, P, b); }

  private:
    LU_type                        LU;
    mtl::dense_vector<size_type>   P;
};


template <typename Matrix, typename Factorizer, typename Value, typename Vector>
struct ilu_solver
  : mtl::vector::assigner<ilu_solver<Matrix, Factorizer, Value, Vector> >
{
    typedef ilu<Matrix, Factorizer, Value> pc_type;

    ilu_solver(const ilu<Matrix, Factorizer, Value>& P, const Vector& x) : P(P), x(x) {}

    template <typename VectorOut>
    void assign_to(VectorOut& y) const
    {	P.solve(x, y);    }    

    const ilu<Matrix, Factorizer, Value>& P; 
    const Vector&                         x;
};


/// Solve LU x = b --> x= U^{-1} L^{-1} b
template <typename Matrix, typename Factorizer, typename Value, typename Vector>
ilu_solver<Matrix, Factorizer, Value, Vector> solve(const ilu<Matrix, Factorizer, Value>& P, const Vector& x)
{
    return ilu_solver<Matrix, Factorizer, Value, Vector>(P, x);
}


/// Solve (LU)^H x = b --> x= L^{-H} U^{-H} b
template <typename Matrix, typename Factorizer, typename Value, typename Vector>
Vector adjoint_solve(const ilu<Matrix, Factorizer, Value>& P, const Vector& b)
{
    return P.adjoint_solve(b);
}

// ic_0_evaluator not needed IC(0) and ILU(0) do the same at the upper triangle ;-)

}} // namespace itl::pc


#endif // ITL_PC_ILU_INCLUDE
