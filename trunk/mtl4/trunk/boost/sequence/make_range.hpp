// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP
# define BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP

# include <boost/sequence/range.hpp>

namespace boost { namespace sequence { 

template <class Elements, class Begin, class End>
range<Elements,Begin,End> make_range(
    Elements const& m, Begin const& b, End const& e)
{
    return range<Elements,Begin,End>(m,b,e);
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP
