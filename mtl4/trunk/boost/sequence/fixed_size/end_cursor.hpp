// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_END_CURSOR_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_END_CURSOR_DWA200555_HPP

# include <boost/sequence/fixed_size/is_fixed_size.hpp>
# include <boost/sequence/end_cursor_fwd.hpp>
# include <boost/sequence/fixed_size/cursor.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/array.hpp>

namespace boost { namespace sequence {

namespace end_cursor_
{
  template <class T, std::size_t N>
  struct implementation<T[N]>
  {
      typedef fixed_size::cursor<N> type;
  };

  template <class T, std::size_t N>
  struct implementation<boost::array<T,N> >
  {
      typedef fixed_size::cursor<N> type;
  };
}

}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_END_CURSOR_DWA200555_HPP
