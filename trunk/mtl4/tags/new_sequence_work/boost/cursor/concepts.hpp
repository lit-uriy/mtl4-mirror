// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_CURSOR_CONCEPTS_DWA2006919_HPP
# define BOOST_CURSOR_CONCEPTS_DWA2006919_HPP

# include <boost/iterator/iterator_concepts.hpp>
# include <boost/iterator/iterator_traits.hpp>
# include <boost/concept_check.hpp>
# include <boost/cursor/next.hpp>
# include <boost/cursor/prior.hpp>

namespace boost { namespace cursor { namespace concepts {

using namespace boost_concepts;

template <class C>
struct Cursor
  : Assignable<C>
  , CopyConstructible<C>
  , EqualityComparable<C>
{};

}}} // namespace boost::cursor

#endif // BOOST_CURSOR_CONCEPTS_DWA2006919_HPP
