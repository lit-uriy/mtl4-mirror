// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP

# include <boost/sequence/intrinsic/tag_fwd.hpp>
# include <cstddef>
# include <boost/array.hpp>

namespace boost { namespace sequence {

namespace fixed_size
{
  // for all fixed-sized sequences S of size N,
  // intrinsic::tag<S>::type should be fixed_size::tag<N>
  template <std::size_t N>
  struct tag {};
}

namespace intrinsic
{
  template <class T, std::size_t N>
  struct tag_impl<T[N]>
  {
      typedef fixed_size::tag<N> type;
  };

  template <class T, std::size_t N>
  struct tag_impl<array<T,N> >
  {
      typedef fixed_size::tag<N> type;
  };
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP
