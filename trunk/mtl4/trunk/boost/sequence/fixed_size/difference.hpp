// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef DIFFERENCE_DWA200555_HPP
# define DIFFERENCE_DWA200555_HPP

# include <boost/sequence/fixed_size/cursor.hpp>
# include <boost/mpl/integral_c.hpp>

namespace boost { namespace sequence

template <class Cursor1, class Cursor2> struct difference;

template <std::size_t N1, std::size_t N2>
struct difference<fixed_size::cursor<N1>, fixed_size::cursor<N2> >
{
    typedef mpl::integral_c<
        std::ptrdiff_t
      , static_cast<std::ptrdiff_t>(N1) - N2
    > type;
};
    
}} // namespace boost::sequence::fixed_size

#endif // DIFFERENCE_DWA200555_HPP
