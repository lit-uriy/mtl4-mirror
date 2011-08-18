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

#ifndef ITL_PC_IC_0_INCLUDE
#define ITL_PC_IC_0_INCLUDE

#include <boost/mpl/bool.hpp>

#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/linear_algebra/inverse.hpp>

#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/operation/lower_trisolve.hpp>
#include <boost/numeric/mtl/operation/upper_trisolve.hpp>
#include <boost/numeric/mtl/matrix/upper.hpp>
#include <boost/numeric/mtl/matrix/strict_lower.hpp>
#include <boost/numeric/mtl/matrix/compressed2D.hpp>
#include <boost/numeric/mtl/matrix/parameter.hpp>
#include <boost/numeric/mtl/matrix/transposed_view.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>
#include <boost/numeric/mtl/vector/assigner.hpp>


namespace itl { namespace pc {

template <typename Matrix>
class ic_0
{
  public:
    typedef typename mtl::Collection<Matrix>::value_type             value_type;
    typedef typename mtl::Collection<Matrix>::size_type              size_type;
    typedef ic_0                                                     self;

    typedef mtl::matrix::parameters<mtl::row_major, mtl::index::c_index, mtl::non_fixed::dimensions, false, size_type> para;
    typedef mtl::compressed2D<value_type, para>                      U_type;
#ifndef ITL_IC_0_ONE_MATRIX
    typedef U_type                                                   L_type;
#else
    typedef typename mtl::matrix::transposed_view<U_type>            L_type;
#endif
    typedef mtl::matrix::detail::lower_trisolve_t<L_type, mtl::tag::inverse_diagonal, true> lower_solver_t;
    typedef mtl::matrix::detail::upper_trisolve_t<U_type, mtl::tag::inverse_diagonal, true> upper_solver_t;

    ic_0(const Matrix& A) : f(A, U), L(trans(U)), lower_solver(L), upper_solver(U) {}


    // solve x = U^T U y --> y= U^{-1} U^{-T} x
    template <typename Vector>
    Vector solve(const Vector& x) const
    {
	mtl::vampir_trace<5036> tracer;
	return inverse_upper_trisolve(U, inverse_lower_trisolve(adjoint(U), x));
    }

    // solve x = U^T U y --> y= U^{-1} U^{-T} x
    template <typename VectorIn, typename VectorOut>
    void solve(const VectorIn& x, VectorOut& y) const
    {
	mtl::vampir_trace<5037> tracer;
	static VectorOut y0(resource(y));
	y.checked_change_resource(x);

	lower_solver(x, y0);
	upper_solver(y0, y);
    }

    // solve x = (LU)^T y --> y= L^{-T} U^{-T} x
    template <typename Vector>
    Vector adjoint_solve(const Vector& x) const
    {
	mtl::vampir_trace<5044> tracer;
	return solve(x);
    }


    L_type get_L() { return L_type(L); }
    U_type get_U() { return U; }

  protected:


    // Dummy type to perform factorization in initializer list not in 
    struct factorizer
    {
	factorizer(const Matrix &A, U_type& U)
	{   factorize(A, U, mtl::traits::is_sparse<Matrix>());  }

	void factorize(const Matrix&, U_type&, boost::mpl::false_)
	{
	    MTL_THROW_IF(true, mtl::logic_error("IC(0) is not suited for dense matrices"));
	}

	// Factorization adapted from Saad
	// Undefined if matrix is not symmetric 
	void factorize(const Matrix& A, U_type& U, boost::mpl::true_)
	{
	    using namespace mtl; using namespace mtl::tag;  using mtl::traits::range_generator;  
	    using math::reciprocal; using mtl::matrix::upper;
	    mtl::vampir_trace<5035> tracer;

	    typedef typename range_generator<row, U_type>::type       cur_type;    
	    typedef typename range_generator<nz, cur_type>::type      icur_type;            

	    MTL_THROW_IF(num_rows(A) != num_cols(A), mtl::matrix_not_square());
	    U= upper(A);

	    typename mtl::traits::col<U_type>::type                   col(U);
	    typename mtl::traits::value<U_type>::type                 value(U); 	

	    cur_type kc= begin<row>(U), kend= end<row>(U);
	    for (size_type k= 0; kc != kend; ++kc, ++k) {

		icur_type ic= begin<nz>(kc), iend= end<nz>(kc);
		MTL_DEBUG_THROW_IF(col(*ic) != k, mtl::missing_diagonal());

		// U[k][k]= 1.0 / sqrt(U[k][k]);
		value_type inv_dia= reciprocal(sqrt(value(*ic)));
		value(*ic, inv_dia);
		icur_type jbegin= ++ic;
		for (; ic != iend; ++ic) {
		    // U[k][i] *= U[k][k]
		    value_type d= value(*ic) * inv_dia;
		    value(*ic, d);
		    size_type i= col(*ic);

		    // find non-zeros U[j][i] below U[k][i] for j in (k, i]
		    // 1. Go to ith row in U (== ith column in U)
		    cur_type irow(i, U); // = begin<row>(U); irow+= i;
		    // 2. Find nonzeros with col() in (k, i]
		    icur_type jc= begin<nz>(irow), jend= end<nz>(irow);
		    while (col(*jc) <= k)  ++jc;
		    while (col(*--jend) > i) ;
		    ++jend; 
		
		    for (; jc != jend; ++jc) {
			size_type j= col(*jc);
			U.lvalue(j, i)-= d * U[k][j];
		    }
		    // std::cout << "U after eliminating U[" << i << "][" << k << "] =\n" << U;
		}
	    }
	}
    };

    U_type                       U;
    factorizer                   f;
    L_type                       L;
    lower_solver_t               lower_solver;
    upper_solver_t               upper_solver;
}; 

template <typename Matrix, typename Vector>
struct ic_0_solver
  : mtl::vector::assigner<ic_0_solver<Matrix, Vector> >
{
    ic_0_solver(const ic_0<Matrix>& P, const Vector& x) : P(P), x(x) {}

    template <typename VectorOut>
    void assign_to(VectorOut& y) const
    {	P.solve(x, y);    }    

    const ic_0<Matrix>& P; 
    const Vector&       x;
};

template <typename Matrix, typename Vector>
ic_0_solver<Matrix, Vector> solve(const ic_0<Matrix>& P, const Vector& x)
{
    return ic_0_solver<Matrix, Vector>(P, x);
}

template <typename Matrix, typename Vector>
Vector adjoint_solve(const ic_0<Matrix>& P, const Vector& x)
{
    return P.adjoint_solve(x);
}


}} // namespace itl::pc

#endif // ITL_PC_IC_0_INCLUDE
