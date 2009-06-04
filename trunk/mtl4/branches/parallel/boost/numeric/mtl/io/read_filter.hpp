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

#ifndef MTL_IO_READ_FILTER_INCLUDE
#define MTL_IO_READ_FILTER_INCLUDE

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/matrix/distributed.hpp>

namespace mtl { namespace io {

/// Utility to filter entries in the read process
/** Depending on type only certain entries are considered for insertion.
    Particularly interesting for distributed collections (inserters). **/
template <typename Inserter>
class read_filter
{
    read_filter() {} // delete
  public:
    explicit read_filter(const Inserter&) {}
    
    /// Default for vectors is to consider every entry
    bool operator()(std::size_t) const { return true; }

    /// Default for matrices is to consider every entry
    bool operator()(std::size_t, std::size_t) const { return true; }
};

/// Specialization for distributed matrix insertion
template <typename Matrix, typename RowDist, typename ColDist, typename Updater>
class read_filter<mtl::matrix::inserter<mtl::matrix::distributed<Matrix, RowDist, ColDist>, Updater> >
{
    typedef mtl::matrix::inserter<mtl::matrix::distributed<Matrix, RowDist, ColDist>, Updater> inserter_type;
    read_filter() {} // delete
  public:
    explicit read_filter(const inserter_type& inserter) : row_dist(row_distribution(reference(inserter))) {}
    
    /// Default for matrices is to consider every entry
    bool operator()(std::size_t r, std::size_t) const { return row_dist.is_local(r); }

  private:
    const RowDist& row_dist;
};


}} // namespace mtl::io

#endif // MTL_IO_READ_FILTER_INCLUDE
