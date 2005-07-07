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
struct tag_impl<T[N]>
{
    typedef fixed_size_indexable_tag<N> type;
};

template <class T, std::size_t N>
struct tag_impl<array<T,N> >
{
    typedef fixed_size_indexable_tag<N> type;
};

}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_TAG_DWA2005617_HPP
