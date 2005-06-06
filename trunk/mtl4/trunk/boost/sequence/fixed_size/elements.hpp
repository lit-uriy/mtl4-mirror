// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP

# include <boost/sequence/index_property_map.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence {

// Don't need two overloads, since there are no array rvalues.
template <class T, std::size_t N>
inline index_property_map<T(&)[N]>
elements(T(&s)[N])
{
    return index_property_map<T(&)[N]>(s);
}

template <class T, std::size_t N>
inline index_property_map<array<T,N>&>
elements(array<T,N>& s)
{
    return index_property_map<array<T,N>&>(s);
}

template <class T, std::size_t N>
inline index_property_map<array<T,N> const&>
elements(array<T,N> const& s)
{
    return index_property_map<array<T,N> const&>(s);
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_FIXED_SIZE_ELEMENTS_DWA200555_HPP
