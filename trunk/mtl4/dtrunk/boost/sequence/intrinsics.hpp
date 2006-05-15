// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSICS_DWA2006513_HPP
# define BOOST_SEQUENCE_INTRINSICS_DWA2006513_HPP

# include <boost/sequence/tag.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class Seq, class Tag = typename tag<Seq>::type>
  struct intrinsics;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INTRINSICS_DWA2006513_HPP
