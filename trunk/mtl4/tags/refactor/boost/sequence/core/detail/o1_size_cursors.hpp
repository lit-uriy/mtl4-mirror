// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_O1_SIZE_CURSORS_DWA20051129_HPP
# define BOOST_SEQUENCE_DETAIL_O1_SIZE_CURSORS_DWA20051129_HPP

# include <boost/sequence/core/detail/is_random_access_cursor.hpp>

# include <boost/type_traits/is_same.hpp>

# include <boost/mpl/and.hpp>

namespace boost { namespace sequence { namespace detail { 

// true iff O(1) size is available from the given pair of cursors
template <class Begin, class End>
struct o1_size_cursors
  : mpl::and_<
        is_same<Begin,End>
      , is_random_access_cursor<Begin>
    >
{};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_O1_SIZE_CURSORS_DWA20051129_HPP
