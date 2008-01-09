// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_MULT_SPECIALIZE_INCLUDE
#define MTL_MULT_SPECIALIZE_INCLUDE

#include <boost/numeric/mtl/operation/dmat_dmat_mult.hpp>
#include <boost/mpl/bool.hpp>

namespace mtl { namespace detail {

template <typename MatrixA, typename MatrixB, typename MatrixC>
struct dmat_dmat_mult_tiling1
{
    static const unsigned long value= 4;
};

template <typename MatrixA, typename MatrixB, typename MatrixC>
struct dmat_dmat_mult_tiling2
{
    static const unsigned long value= 4;
};

template <typename MatrixA, typename MatrixB, typename MatrixC>
struct dmat_dmat_mult_specialize
    : public boost::mpl::false_
{};

/*
   In order to specialize the functor, write for instance:

template <typename MatrixA, typename MatrixB, typename MatrixC>
struct dmat_dmat_mult_specialize
    : public boost::mpl::true_
{
    typedef gen_dmat_dmat_mult_t<> type;
};
*/




}} // namespace mtl::detail

#endif // MTL_MULT_SPECIALIZE_INCLUDE
