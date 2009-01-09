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

#ifndef MTL_TRAITS_EVAL_INCLUDE
#define MTL_TRAITS_EVAL_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/matrix/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>

namespace mtl { namespace traits {


template <typename T>
struct eval {};

#if 0 // To be done later
template <typename Value, typename Parameter>
struct eval< mtl::vector::dense_vector<Value, Parameter> >
{};


template <typename Value1, typename Vector>
struct eval< mtl::vector::scaled_view<Value1, Vector> > 
{};

template <typename Value1, typename Vector>
struct eval< mtl::vector::rscaled_view<Value1, Vector> > 
{};
#endif


    namespace impl {
	
	template<typename Matrix, typename ValueType, typename SizeType>
	struct eval_matrix_ref 
	    : public mtl::matrix::const_crtp_base_matrix<Matrix, ValueType, SizeType>,
	      public mtl::matrix::mat_expr< eval_matrix_ref< Matrix, ValueType, SizeType > >
	{
	    eval_matrix_ref(const Matrix& ref) : ref(ref) {}

	    ValueType operator()(SizeType r, SizeType c) const { return ref[r][c]; }	    

	    const Matrix& ref;
	};

	template<typename Matrix, typename ValueType, typename SizeType>
	inline SizeType num_rows(const eval_matrix_ref< Matrix, ValueType, SizeType >& ev) { return num_rows(ev.ref); }

	template<typename Matrix, typename ValueType, typename SizeType>
	inline SizeType num_cols(const eval_matrix_ref< Matrix, ValueType, SizeType >& ev) { return num_cols(ev.ref); }

	template<typename Matrix, typename ValueType, typename SizeType>
	inline SizeType size(const eval_matrix_ref< Matrix, ValueType, SizeType >& ev) { return size(ev.ref); }
    }


template <typename Value, typename Parameter>
struct eval< mtl::matrix::dense2D<Value, Parameter> >
// : public impl::eval_matrix_ref< mtl::matrix::dense2D<Value, Parameter>, Value, std::size_t >
{
    typedef mtl::matrix::dense2D<Value, Parameter>  matrix_type;

    eval(const matrix_type& ref) : ref(ref) {}

    const matrix_type& value() const { return ref; }

#if 0
    eval(const mtl::matrix::dense2D<Value, Parameter>& ref)
	: impl::eval_matrix_ref< mtl::matrix::dense2D<Value, Parameter>, Value, std::size_t >(ref)
    {}
#endif
private:
    const matrix_type& ref;
};

template <typename Value, long unsigned Mask, typename Parameter>
struct eval< mtl::matrix::morton_dense<Value, Mask, Parameter> >
{};

template <typename Value, typename Parameter>
struct eval< mtl::matrix::compressed2D<Value, Parameter> >
{};





template <typename E1, typename E2>
struct eval< mtl::matrix::mat_mat_asgn_expr<E1, E2> > 
{};

template <typename E1, typename E2>
struct eval< mtl::matrix::mat_mat_plus_expr<E1, E2> > 
{};

template <typename E1, typename E2>
struct eval< mtl::matrix::mat_mat_minus_expr<E1, E2> > 
{};

template <typename E1, typename E2>
struct eval< mtl::matrix::mat_mat_ele_times_expr<E1, E2> > 
    : boost::mpl::bool_< eval<E1>::value && eval<E1>::value >
{};

template <typename Value1, typename Matrix>
struct eval< mtl::matrix::scaled_view<Value1, Matrix> > 
{};

template <typename Value1, typename Matrix>
struct eval< mtl::matrix::rscaled_view<Value1, Matrix> > 
{};


}} // namespace mtl::traits

#endif // MTL_TRAITS_EVAL_INCLUDE
