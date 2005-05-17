// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ADVANCED_DWA200555_HPP
# define ADVANCED_DWA200555_HPP

# include <boost/sequence/fixed_size/cursor.hpp>

namespace boost { namespace sequence

template <class Cursor, class Amount> struct advanced;

template <std::size_t N, class Amount>
struct advanced<fixed_size::cursor<N>, Amount>
{
    typedef fixed_size::cursor<N + Amount::value> type;
};
    
}} // namespace boost::sequence::fixed_size

#endif // ADVANCED_DWA200555_HPP
