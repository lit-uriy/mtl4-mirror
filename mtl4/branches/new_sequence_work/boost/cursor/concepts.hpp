// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_CURSOR_CONCEPTS_DWA2006919_HPP
# define BOOST_CURSOR_CONCEPTS_DWA2006919_HPP

# include <boost/iterator/iterator_concepts.hpp>
# include <boost/iterator/iterator_traits.hpp>
# include <boost/concept_check.hpp>
# include <boost/cursor/dereferenceable.hpp>
# include <boost/cursor/next.hpp>
# include <boost/cursor/prior.hpp>
# include <boost/detail/is_incrementable.hpp>
# include <boost/mpl/assert.hpp>

namespace boost { namespace cursor { namespace concepts {

using namespace boost_concepts;

template <class C>
struct Cursor
  : Assignable<C>
  , CopyConstructible<C>
  , EqualityComparable<C>
{
    typedef typename
      boost::detail::is_incrementable<C>::type
    has_preincrement;
};

template <class C>
struct DereferenceableCursor
  : Cursor<C>
{
    typedef typename result_of<op::deref(C)>::type key_type;
        
    BOOST_CONCEPT_USAGE(DereferenceableCursor)
    {
        key_type k = cursor::deref(p);
    }
 private:
    Cursor const p;
};

// A Cursor that can be incremented in-place
template <class C>
struct DynamicCursor
  : DereferenceableCursor<C>
{
    BOOST_CONCEPT_USAGE(DynamicCursor)
    {
        Cursor p2 = cursor::next(p1);
    }
 private:
    Cursor const p1;
};

namespace detail
{
  struct empty {};
}

template <class C>
struct TraversibleCursor
  : Cursor<C>
  , mpl::if_<
        typename Cursor<C>::has_preincrement
      , DynamicCursor<C>
      , empty
    >
{
};



}}} // namespace boost::cursor

#endif // BOOST_CURSOR_CONCEPTS_DWA2006919_HPP
