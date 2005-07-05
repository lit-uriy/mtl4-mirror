// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP

# include <boost/sequence/tag_fwd.hpp>
# include <cstddef>
# include <boost/array.hpp>

namespace boost { namespace sequence { 

template <std::size_t N>
struct fixed_size_random_access_tag {};

template <std::size_t N>
struct fixed_size_indexable_tag
  : fixed_size_random_access_tag<N>
{};

template <class T, std::size_t N>
struct tag_base<T[N]>
{
    typedef fixed_size_indexable_tag<N> type;
};

template <class T, std::size_t N>
struct tag_base<array<T,N> >
{
    typedef fixed_size_indexable_tag<N> type;
};

# if BOOST_WORKAROUND(_MSC_FULL_VER, <= 140050215)

// Jason Shirk assures me this bug is fixed for the release version of
// VC++ 8.0.  I'm not sure the workaround is much help since other
// array-related confusions break VC++ 7.1 and VC++ 8.0 beta when used
// on built-in arrays.
template <class T>
struct array_tag
{
    static T& make();
    typedef fixed_size_indexable_tag<(sizeof(T)/sizeof(make()[0]))> type;
};

# endif 


}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP
