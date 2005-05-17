// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP

# include <boost/utility/enable_if.hpp>
# include <boost/fixed_size/is_fixed_size.hpp>
# include <boost/fixed_size/accessor.hpp>

namespace boost { namespace sequence {

template <class S>
typename lazy_enable_if<
    fixed_size::is_fixed_size<S>
  , accessor<S>
>::type
inline elements(S& s)
{
    return accessor<S>::type(s);
}

template <class S>
typename lazy_enable_if<
    fixed_size::is_fixed_size<S>
  , accessor<S const>
>::type
inline elements(S const& s)
{
    return accessor<S const>::type(s);
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
