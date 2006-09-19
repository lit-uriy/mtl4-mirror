// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_CURSOR_TAG_DWA2006513_HPP
# define BOOST_CURSOR_TAG_DWA2006513_HPP

# include <boost/detail/is_incrementable.hpp>

namespace boost { namespace cursor { 

namespace impl
{
  struct homogeneous_tag {};
  struct heterogeneous_tag {};
  
  template <class Cursor>
  struct tag
    : mpl::if_<
          boost::detail::is_incrementable<Cursor>
        , homogeneous_tag
        , heterogeneous_tag
      >
  {};

  template <class Cursor>
  struct tag<Cursor const>
    : tag<Cursor>
  {};
  
  template <class Cursor>
  struct tag<Cursor volatile>
    : tag<Cursor>
  {};
  
  template <class Cursor>
  struct tag<Cursor const volatile>
    : tag<Cursor>
  {};
}

}} // namespace boost::cursor

#endif // BOOST_CURSOR_TAG_DWA2006513_HPP
