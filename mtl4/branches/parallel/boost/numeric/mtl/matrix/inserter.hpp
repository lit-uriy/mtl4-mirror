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

#ifndef MTL_MATRIX_INSERTER_INCLUDE
#define MTL_MATRIX_INSERTER_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/detail/trivial_inserter.hpp>


namespace mtl { namespace matrix {

template <typename Matrix, 
	  typename Updater>
struct inserter 
  : public mtl::detail::trivial_inserter<Matrix, Updater>
{
    typedef mtl::detail::trivial_inserter<Matrix, Updater>     base;
    typedef typename Matrix::size_type   size_type;

    explicit inserter(Matrix& matrix, size_type slot_size = 0) : base(matrix, slot_size) {}
};


template <typename Elt, typename Parameters, typename Updater>
struct inserter<compressed2D<Elt, Parameters>, Updater>
  : public compressed2D_inserter<Elt, Parameters, Updater>
{
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef compressed2D_inserter<Elt, Parameters, Updater > base;

    explicit inserter(matrix_type& matrix, size_type slot_size = 5) : base(matrix, slot_size) {}
};


template <typename Matrix, typename RowDistribution, typename ColDistribution, typename Updater>
struct inserter<distributed<Matrix, RowDistribution, ColDistribution>, Updater>
    : public distributed_inserter<distributed<Matrix, RowDistribution, ColDistribution>, Updater>
{
    typedef distributed<Matrix, RowDistribution, ColDistribution>                                  matrix_type;
    typedef typename Collection<matrix_type>::size_type                                            size_type;
    typedef distributed_inserter<distributed<Matrix, RowDistribution, ColDistribution>, Updater>   base;

    explicit inserter(matrix_type& matrix, size_type slot_size = 5) : base(matrix, slot_size) {}
};


}} // namespace mtl::matrix

#endif // MTL_MATRIX_INSERTER_INCLUDE
