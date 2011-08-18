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

#ifndef MTL_UPPER_TRISOLVE_INCLUDE
#define MTL_UPPER_TRISOLVE_INCLUDE

#include <boost/mpl/int.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/operation/resource.hpp>
#include <boost/numeric/linear_algebra/identity.hpp>
#include <boost/numeric/mtl/interface/vpt.hpp>


namespace mtl { namespace matrix {


namespace detail {

    // CompactStorage means that matrix contains only upper entries (strict upper when DiaTag == unit_diagonal)
    template <typename Matrix, typename DiaTag, bool CompactStorage= false>
    struct upper_trisolve_t
    {
	typedef typename Collection<Matrix>::value_type           value_type;
	typedef typename Collection<Matrix>::size_type            size_type;
	typedef typename OrientedCollection<Matrix>::orientation  my_orientation;
	typedef typename mtl::traits::category<Matrix>::type      my_category;
	typedef typename mtl::traits::range_generator<tag::major, Matrix>::type     a_cur_type; // row or col depending on Matrix    
	typedef typename mtl::traits::range_generator<tag::nz, a_cur_type>::type    a_icur_type;   

	upper_trisolve_t(const Matrix& A) : A(A), value_a(A), col_a(A), row_a(A)
	{    MTL_THROW_IF(num_rows(A) != num_cols(A), matrix_not_square());	}

	template <typename M, typename D, bool C>
	struct generic_version
	  : boost::mpl::int_<mtl::traits::is_row_major<M>::value ? 1 : 2> {};

	template <typename M, typename D, bool C>
	struct version
	  : generic_version<M, D, C> {};

#if 1	
	template <typename Value, typename Para, typename D>
	struct version<compressed2D<Value, Para>, D, true>
	  : boost::mpl::if_<mtl::traits::is_row_major<Para>,
			    boost::mpl::int_<3>,
			    generic_version<compressed2D<Value, Para>, D, true>
	                   >::type {};
#endif
	template <typename VectorIn, typename VectorOut>
	void operator()(const VectorIn& v, VectorOut& w) const
	{
	    apply(v, w, version<Matrix, DiaTag, CompactStorage>());
	}

	template <typename Vector>
	Vector operator()(const Vector& v) const
	{
	    Vector w(resource(v));
	    (*this)(v, w);
	    return w;
	}

    private:
	// Initialization for regular and inverse diagonal is the same
	template <typename Cursor>
	void row_init(size_type MTL_DEBUG_ARG(r), Cursor& aic, Cursor& MTL_DEBUG_ARG(aiend), value_type& dia, tag::universe_diagonal) const
	{
	    MTL_DEBUG_THROW_IF(aic == aiend || col_a(*aic) != r, missing_diagonal());
	    dia= value_a(*aic); ++aic;
	}

	template <typename Cursor>
	void row_init(size_type, Cursor&, Cursor&, value_type&, tag::unit_diagonal) const {}

	void row_update(value_type& res, value_type& rr, const value_type& dia, tag::regular_diagonal) const { res= rr / dia; }
	void row_update(value_type& res, value_type& rr, const value_type& dia, tag::inverse_diagonal) const { res= rr * dia;	}
	void row_update(value_type& res, value_type& rr, const value_type&    , tag::unit_diagonal)    const { res= rr; }

	template <typename Tag> int dia_inc(Tag) const { return 0; }
	int dia_inc(tag::unit_diagonal) const { return 1; }

	// Generic row-major
	template <typename VectorIn, typename VectorOut>
	void inline apply(const VectorIn& v, VectorOut& w, boost::mpl::int_<1>) const
	{
	    vampir_trace<5042> tracer;
	    using namespace tag; 
	    a_cur_type ac= begin<row>(A), aend= end<row>(A); 
	    for (size_type r= num_rows(A) - 1; ac != aend--; --r) {
		// std::cout << "row " << r << '\n';
		a_icur_type aic= CompactStorage ? begin<nz>(aend) : lower_bound<nz>(aend, r + dia_inc(DiaTag())), 
		            aiend= end<nz>(aend);
		value_type rr= v[r], dia;
		// std::cout << "rr[init] " << rr << '\n';
		row_init(r, aic, aiend, dia, DiaTag()); 
		for (; aic != aiend; ++aic) {
		    // std::cout << ", column " << col_a(*aic) << '\n';
		    MTL_DEBUG_THROW_IF(col_a(*aic) <= r, logic_error("Matrix entries must be sorted for this."));
		    rr-= value_a(*aic) * w[col_a(*aic)];
		    // std::cout << "dec " << value_a(*aic) << " * " << w[col_a(*aic)] << '\n';
		    // std::cout << "rr " << rr << '\n';
		}
		row_update(w[r], rr, dia, DiaTag());
		// std::cout << "w[r] " << w[r] << '\n';
	    }
	}

	// Generic column-major
	template <typename VectorIn, typename VectorOut>
	void apply(const VectorIn& v, VectorOut& w, boost::mpl::int_<2>) const
	{
	    vampir_trace<5043> tracer;
	    using namespace tag; 
	    w= v;
	    a_cur_type ac= begin<col>(A), aend= end<col>(A); 
	    for (size_type r= num_rows(A) - 1; ac != aend--; --r) {
		a_icur_type aic= begin<nz>(aend), 
		            aiend= CompactStorage ? end<nz>(aend) : lower_bound<nz>(aend, r + 1 - dia_inc(DiaTag()));
		value_type rr;
		col_init(r, aic, aiend, rr, w[r], DiaTag());

		for (; aic != aiend; ++aic) {
		    MTL_DEBUG_THROW_IF(row_a(*aic) >= r, logic_error("Matrix entries must be sorted for this."));
		    w[row_a(*aic)]-= value_a(*aic) * rr;
		}
	    }
	}

	void crs_row_init(size_type MTL_DEBUG_ARG(r), size_type& j0, size_type MTL_DEBUG_ARG(cj1), value_type& dia, tag::universe_diagonal) const
	{
	    MTL_DEBUG_THROW_IF(j0 == cj1 || A.ref_indices()[j0] != r, missing_diagonal());
	    dia= A.data[j0++];
	}
	void crs_row_init(size_type, size_type&, size_type, value_type&, tag::unit_diagonal) const {}

	// Tuning for IC_0 and similar at compressed2D row-major compact
	template <typename VectorIn, typename VectorOut>
	void apply(const VectorIn& v, VectorOut& w, boost::mpl::int_<3>) const
	{
	    vampir_trace<5046> tracer;
	    // std::cout << "In getuneter Version\n";
	    for (size_type r= num_rows(A); r-- > 0; ) {
		// std::cout << "row " << r << '\n';
		size_type j0= A.ref_starts()[r];
		const size_type cj1= A.ref_starts()[r+1];
		value_type rr= v[r], dia;
		// std::cout << "rr[init] " << rr << '\n';
		crs_row_init(r, j0, cj1, dia, DiaTag()); 
		// std::cout << "dia " << dia << '\n';
		for (; j0 != cj1; ++j0) {
		    // std::cout << "offset " << j0 << ", column " << A.ref_indices()[j0] << '\n';
		    MTL_DEBUG_THROW_IF(A.ref_indices()[j0] <= r, logic_error("Matrix entries must be sorted for this."));
		    rr-= A.data[j0] * w[A.ref_indices()[j0]];
		    // std::cout << "dec " << A.data[j0] << " * " << w[A.ref_indices()[j0]] << '\n';
		    // std::cout << "rr " << rr << '\n';
		}
		row_update(w[r], rr, dia, DiaTag());
		// std::cout << "w[r] " << w[r] << '\n';
	    }
	}

	template <typename Cursor>
	void col_init(size_type MTL_DEBUG_ARG(r), Cursor& MTL_DEBUG_ARG(aic), Cursor& aiend, value_type& rr, value_type& res, tag::regular_diagonal) const
	{
	    MTL_DEBUG_THROW_IF(aic == aiend, missing_diagonal());
	    --aiend;
	    MTL_DEBUG_THROW_IF(row_a(*aiend) != r, missing_diagonal());
	    rr= res/= value_a(*aiend);
	}
	
	template <typename Cursor>
	void col_init(size_type MTL_DEBUG_ARG(r), Cursor& MTL_DEBUG_ARG(aic), Cursor& aiend, value_type& rr, value_type& res, tag::inverse_diagonal) const
	{
	    MTL_DEBUG_THROW_IF(aic == aiend, missing_diagonal());
	    --aiend;
	    MTL_DEBUG_THROW_IF(row_a(*aiend) != r, missing_diagonal());
	    rr= res*= value_a(*aiend);
	}

	template <typename Cursor>
	void col_init(size_type, Cursor&, Cursor&, value_type& rr, value_type& res, tag::unit_diagonal) const
	{
	    rr= res;
	}


	const Matrix& A;
	typename mtl::traits::const_value<Matrix>::type  value_a; 
	typename mtl::traits::col<Matrix>::type          col_a; 
	typename mtl::traits::row<Matrix>::type          row_a;
    };

}

/// Solves the upper triangular matrix A  with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline upper_trisolve(const Matrix& A, const Vector& v)
{
    vampir_trace<3043> tracer;
    return detail::upper_trisolve_t<Matrix, tag::regular_diagonal>(A)(v);
}

/// Solves the upper triangular matrix A  with the rhs v while solution vector w is passed as reference
template <typename Matrix, typename VectorIn, typename VectorOut>
void inline upper_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    vampir_trace<3043> tracer;
    detail::upper_trisolve_t<Matrix, tag::regular_diagonal> solver(A); // use of anonymous variable causes weird error
    solver(v, w);
}

/// Solves the upper triangular matrix A (only one's in the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline unit_upper_trisolve(const Matrix& A, const Vector& v)
{
    vampir_trace<3044> tracer;
    return detail::upper_trisolve_t<Matrix, tag::unit_diagonal>(A)(v);
}

/// Solves the upper triangular matrix A (only one's in the diagonal) with the rhs v while solution vector w is passed as reference
template <typename Matrix, typename VectorIn, typename VectorOut>
void inline unit_upper_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    vampir_trace<3044> tracer;
    detail::upper_trisolve_t<Matrix, tag::unit_diagonal> solver(A);
    solver(v, w);
}

/// Solves the upper triangular matrix A  (inverse the diagonal) with the rhs v and returns the solution vector
template <typename Matrix, typename Vector>
Vector inline inverse_upper_trisolve(const Matrix& A, const Vector& v)
{
    vampir_trace<3045> tracer;
    return detail::upper_trisolve_t<Matrix, tag::inverse_diagonal>(A)(v);
}

/// Solves the upper triangular matrix A  (inverse the diagonal) with the rhs v while solution vector w is passed as reference
template <typename Matrix, typename VectorIn, typename VectorOut>
void inline inverse_upper_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w)
{
    vampir_trace<3045> tracer;
    detail::upper_trisolve_t<Matrix, tag::inverse_diagonal> solver(A);
    solver(v, w);
}

/// Solves the upper triangular matrix A  with the rhs v and returns the solution vector
template <typename Matrix, typename Vector, typename DiaTag>
Vector inline upper_trisolve(const Matrix& A, const Vector& v, DiaTag)
{
    vampir_trace<3046> tracer;
    return detail::upper_trisolve_t<Matrix, DiaTag>(A)(v);
}

/// Solves the upper triangular matrix A  with the rhs v while solution vector w is passed as reference
template <typename Matrix, typename VectorIn, typename VectorOut, typename DiaTag>
void inline upper_trisolve(const Matrix& A, const VectorIn& v, VectorOut& w, DiaTag)
{
    vampir_trace<3046> tracer;
    detail::upper_trisolve_t<Matrix, DiaTag> solver(A);
    solver(v, w);
}

}} // namespace mtl::matrix

#endif // MTL_UPPER_TRISOLVE_INCLUDE
