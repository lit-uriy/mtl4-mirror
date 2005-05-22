// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_END_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_END_DWA200555_HPP

# include <boost/utility/enable_if.hpp>
# include <boost/sequence/fixed_size/is_fixed_size.hpp>
# include <boost/sequence/fixed_size/accessor.hpp>

namespace boost { namespace sequence {

template <class S>
typename lazy_enable_if<
    fixed_size::is_fixed_size<S>
  , end_cursor<S>
>::type
inline end(S& s)
{
    return end_cursor<S>::type();
}

template <class S>
typename lazy_enable_if<
    fixed_size::is_fixed_size<S>
  , end_cursor<S const>
>::type
inline end(S const& s)
{
    return end_cursor<S const>::type();
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_END_DWA200555_HPP
