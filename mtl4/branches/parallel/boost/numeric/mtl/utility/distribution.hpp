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

#ifndef MTL_TRAITS_DISTRIBUTION_INCLUDE
#define MTL_TRAITS_DISTRIBUTION_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_same.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/root.hpp>

namespace mtl { namespace traits {

// Forward declarations
template <typename T> struct row_distribution;
// template <typename T> struct col_distribution;

template <typename T> struct distribution_aux {}; // by default empty

/// Type trait for distribution type of containers and expressions
template <typename T>
struct distribution 
  : distribution_aux<typename root<T>::type>
{};

template <typename Vector, typename Distribution>
struct distribution_aux<mtl::vector::distributed<Vector, Distribution> >
{
    typedef Distribution type;
};

template <typename Value, typename Parameters> 
struct distribution_aux<mtl::vector::dense_vector<Value, Parameters> >
{
    typedef par::replication type;
};

template <typename Functor, typename Vector>
struct distribution_aux<mtl::vector::map_view<Functor, Vector> >
  : distribution<Vector>
{};

template <typename E1, typename E2, typename Functor>
struct distribution_aux<mtl::vector::vec_vec_aop_expr<E1, E2, Functor> >
  : distribution<E1>
{
    BOOST_STATIC_ASSERT((boost::is_same<typename distribution<E1>::type, typename distribution<E2>::type>::value));
};

template <typename E1, typename E2, typename Functor>
struct distribution_aux<mtl::vector::vec_vec_pmop_expr<E1, E2, Functor> >
  : distribution<E1>
{
    BOOST_STATIC_ASSERT((boost::is_same<typename distribution<E1>::type, typename distribution<E2>::type>::value));
};

template <typename E1, typename E2, typename Functor>
struct distribution_aux<mtl::vector::vec_scal_aop_expr<E1, E2, Functor> >
  : distribution<E1>
{};

template <typename Matrix, typename CVector>
struct distribution_aux<mtl::mat_cvec_times_expr<Matrix, CVector> >
  : row_distribution<Matrix>
{};


// == Row distribution types ==

template <typename T> struct row_distribution_aux {}; // by default empty

/// Type trait for distribution type of containers and expressions
template <typename T>
struct row_distribution 
  : row_distribution_aux<typename root<T>::type>
{};

template <typename Vector, typename RowDistribution, typename ColDistribution>
struct row_distribution_aux<mtl::matrix::distributed<Vector, RowDistribution, ColDistribution> >
{
    typedef RowDistribution type;
};



}} // namespace mtl::traits

#endif // MTL_TRAITS_DISTRIBUTION_INCLUDE
