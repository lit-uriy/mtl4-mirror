// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ADVANCED_DWA200555_HPP
# define ADVANCED_DWA200555_HPP

# include <boost/sequence/tuple/cursor.hpp>

namespace boost { namespace sequence {

template <class Cursor, class Amount> struct advanced;

template <std::size_t N, class Amount>
struct advanced<tuple::cursor<N>, Amount>
{
    typedef tuple::cursor<N + Amount::value> type;
};
    
template <std::size_t N, class Amount>
struct advanced<tuple::cursor<N> const, Amount>
{
    typedef tuple::cursor<N + Amount::value> type;
};
    
}} // namespace boost::sequence::tuple

#endif // ADVANCED_DWA200555_HPP
