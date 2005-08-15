// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_EXTENT_DWA200559_HPP
# define BOOST_SEQUENCE_EXTENT_DWA200559_HPP

# include <boost/sequence/intrinsics.hpp>

namespace boost { namespace sequence { 

template <class Sequence>
struct extent
  : intrinsic::distance<
        typename intrinsics<Sequence>::begin::type
      , typename intrinsics<Sequence>::end::type
    >
{};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_EXTENT_DWA200559_HPP
