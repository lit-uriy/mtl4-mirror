// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_EXTENT_DWA200559_HPP
# define BOOST_SEQUENCE_EXTENT_DWA200559_HPP

# include <boost/sequence/difference.hpp>
# include <boost/sequence/intrinsics.hpp>

namespace boost { namespace sequence { 

template <class Sequence>
struct extent
  : difference<
        typename intrinsics<Sequence>::end::type
      , typename intrinsics<Sequence>::begin::type
    >
{};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_EXTENT_DWA200559_HPP
