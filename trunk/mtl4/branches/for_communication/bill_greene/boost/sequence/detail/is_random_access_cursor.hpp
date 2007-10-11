// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_IS_RANDOM_ACCESS_CURSOR_DWA20051129_HPP
# define BOOST_SEQUENCE_DETAIL_IS_RANDOM_ACCESS_CURSOR_DWA20051129_HPP

# include <boost/iterator/iterator_categories.hpp>
# include <boost/type_traits/is_convertible.hpp>

namespace boost { namespace sequence { namespace detail {

template <class Cursor>
struct is_random_access_cursor
  : is_convertible<
        typename iterator_traversal<Cursor>::type
      , random_access_traversal_tag
    >
{};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_IS_RANDOM_ACCESS_CURSOR_DWA20051129_HPP
