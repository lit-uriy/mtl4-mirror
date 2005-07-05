// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_TAG_DWA2005616_HPP
# define BOOST_SEQUENCE_TAG_DWA2005616_HPP

# include <boost/sequence/tag_fwd.hpp>
# include <boost/sequence/iterator_range_tag.hpp>
# include <boost/sequence/fixed_size/tag.hpp>
# include <boost/mpl/eval_if.hpp>
# include <boost/type_traits/is_array.hpp>

namespace boost { namespace sequence { 

template <class T>
struct tag_base
{
    typedef iterator_range_tag type;
};


# if BOOST_WORKAROUND(_MSC_FULL_VER, <= 140050215)

// Jason Shirk assures me this bug is fixed for the release version of
// VC++ 8.0.  I'm not sure the workaround is much help since other
// array-related confusions break VC++ 7.1 and VC++ 8.0 beta when used
// on built-in arrays.
template <class T>
struct array_tag;

template <class T>
struct tag
{
    typedef typename mpl::eval_if<is_array<T>,array_tag<T>,tag_base<T> >::type type;
};

# else

template <class T>
struct tag
  : tag_base<T>
{};

# endif 

// The tag for T const is the same as that for T
template < class T >
struct tag< T const >
  : tag<T>
{};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_TAG_DWA2005616_HPP
