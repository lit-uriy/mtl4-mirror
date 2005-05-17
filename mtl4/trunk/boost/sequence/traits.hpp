// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef TRAITS_DWA2005329_HPP
# define TRAITS_DWA2005329_HPP

# include <boost/sequence/array/traits.hpp>
# include <boost/sequence/pointer/traits.hpp>

namespace boost {
namespace sequence { 

// Do not use directly, except to specialize.
// Default implementation works for ordinary sequences.
template <class Sequence>
struct traits
{
    typedef typename Sequence::iterator begin_cursor;
    typedef typename Sequence::iterator end_cursor;
    typedef typename Sequence elements_map;
};
    
template <class Sequence>
struct traits<Sequence const>
{
    typedef typename Sequence::const_iterator begin_cursor;
    typedef typename Sequence::const_iterator end_cursor;
    typedef typename Sequence elements_map;
};

template <class Sequence>
struct begin_cursor
{
    typedef typename sequence::traits<Sequence>::begin_cursor type;
};
    
template <class Sequence>
struct end_cursor
{
    typedef typename sequence::traits<Sequence>::end_iterator type;
};

template <class Sequence>
struct elements_map
{
    typedef typename sequence::traits<Sequence>::end_iterator type;
};

}} // namespace boost::sequence

#endif // TRAITS_DWA2005329_HPP
