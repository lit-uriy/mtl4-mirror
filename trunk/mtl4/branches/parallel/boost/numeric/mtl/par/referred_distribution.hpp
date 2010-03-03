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

#ifndef MTL_MATRIX_REFERRED_DISTRIBUTION_INCLUDE
#define MTL_MATRIX_REFERRED_DISTRIBUTION_INCLUDE

namespace mtl { namespace matrix {

template <typename RowDist>
bool inline referred_distribution_aux(const RowDist& rd, const RowDist& cd)
{
    return &rd == &cd;
}

template <typename RowDist, typename ColDist>
bool inline referred_distribution_aux(const RowDist&, const ColDist&)
{
    return false;
}

/// Helper function determining whether the column distribution refers to row distribution
template <typename Matrix>
bool inline referred_distribution(const Matrix& A)
{
    return referred_distribution_aux(row_distribution(A), col_distribution(A));
}


}} // namespace mtl::matrix

#endif // MTL_MATRIX_REFERRED_DISTRIBUTION_INCLUDE
