// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP
# define BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP

# include <boost/sequence/view/range/range.hpp>
# include <boost/sequence/core/detail/is_mpl_integral_constant.hpp>
# include <boost/mpl/not.hpp>
# include <boost/mpl/or.hpp>
# include <boost/mpl/and.hpp>

namespace boost { namespace sequence { 

template <class Elements, class Begin, class End>
range<Elements,Begin,End> make_range(
    Elements const& m, Begin const& b, End const& e)
{
    return range<Elements,Begin,End>(m,b,e);
}

template <class Elements, class Begin, class End, class Size>
range<Elements,Begin,End> make_range(
    Elements const& m, Begin const& b, End const& e, Size const& s)
{
    return range<Elements,Begin,End,Size>(m,b,e,s);
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_MAKE_RANGE_DWA200559_HPP
