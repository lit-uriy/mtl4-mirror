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

#ifndef MTL_TRAITS_ROOT_INCLUDE
#define MTL_TRAITS_ROOT_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl { namespace traits {


/// Type trait to reduce types to their essentials by removing const, reference, ... and gearing derived types to their bases
template <typename T>
struct root
{
    typedef T        type;
};

// ==========================
// Remove language attributes
// ==========================

template <typename T>
struct root<T&>
{
    typedef T        type;
};

template <typename T>
struct root<const T>
{
    typedef T        type;
};

// ============
// Base classes
// ============

// Implicit dense matrices

template <typename Value>
struct root<mtl::matrix::ones_matrix<Value> >
{
    typedef mtl::matrix::implicit_dense<mtl::matrix::ones_functor<Value> > type;
};

template <typename Value>
struct root<mtl::matrix::hilbert_matrix<Value> >
{
    typedef mtl::matrix::implicit_dense<mtl::matrix::hilbert_functor<Value> > type;
};

template <typename Vector1, typename Vector2>
struct root<mtl::matrix::outer_product_matrix<Vector1, Vector2> >
{
    typedef mtl::matrix::implicit_dense<mtl::matrix::outer_product_functor<Vector1, Vector2> > type;
};

// Matrix map views

template <typename Scaling, typename Matrix>
struct root< mtl::matrix::scaled_view<Scaling, Matrix> >
{
    typedef mtl::matrix::map_view<tfunctor::scale<Scaling, typename Matrix::value_type>, Matrix> type;
};

template <typename Matrix>
struct root< mtl::matrix::conj_view<Matrix> >
{
    typedef mtl::matrix::map_view<sfunctor::conj<typename Matrix::value_type>, Matrix> type;
};

template <typename Matrix, typename RScaling>
struct root< mtl::matrix::rscaled_view<Matrix, RScaling> >
{
    typedef mtl::matrix::map_view<tfunctor::rscale<typename Matrix::value_type, RScaling>, Matrix> type;
};

template <typename Matrix, typename Divisor>
struct root< mtl::matrix::divide_by_view<Matrix, Divisor> >
{
    typedef mtl::matrix::map_view<tfunctor::divide_by<typename Matrix::value_type, Divisor>, Matrix> type;
};

// Vector assignment expressions

template <typename E1, typename E2>
struct root< vector::vec_vec_plus_asgn_expr<E1, E2> >
{
    typedef vector::vec_vec_aop_expr< E1, E2, mtl::sfunctor::plus_assign<typename E1::value_type, typename E2::value_type> > type;
};

template <typename E1, typename E2>
struct root< vector::vec_vec_minus_asgn_expr<E1, E2> >
{
    typedef vector::vec_vec_aop_expr< E1, E2, mtl::sfunctor::minus_assign<typename E1::value_type, typename E2::value_type> > type;
};




#if 0 // template
struct root
{
    typedef  type;
};
#endif


}} // namespace mtl::traits

#endif // MTL_TRAITS_ROOT_INCLUDE
