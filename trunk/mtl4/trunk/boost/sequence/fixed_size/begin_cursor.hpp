// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_BEGIN_CURSOR_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_BEGIN_CURSOR_DWA200555_HPP

# include <boost/sequence/fixed_size/is_fixed_size.hpp>
# include <boost/sequence/begin_cursor_fwd.hpp>
# include <boost/sequence/fixed_size/cursor.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence {

namespace begin_cursor_
{ 
  template <class Sequence>
  struct implementation<
      Sequence
    , typename enable_if<
          fixed_size::is_fixed_size<Sequence>
      >::type
  >
  {
      typedef fixed_size::cursor<0> type;
  };
}

}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_BEGIN_CURSOR_DWA200555_HPP
