// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef PREDECESSOR_DWA200555_HPP
# define PREDECESSOR_DWA200555_HPP

# include <boost/sequence/fixed_size/cursor.hpp>

namespace boost { namespace sequence

template <class Cursor> struct predecessor;

template <std::size_t N>
struct predecessor<fixed_size::cursor<N> >
{
    typedef fixed_size::cursor<N-1> type;
};
    
}} // namespace boost::sequence::fixed_size

#endif // PREDECESSOR_DWA200555_HPP
