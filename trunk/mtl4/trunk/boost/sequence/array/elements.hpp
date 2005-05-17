// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ARRAY_ELEMENTS_DWA200559_HPP
# define BOOST_SEQUENCE_ARRAY_ELEMENTS_DWA200559_HPP

# include <boost/sequence/index_property_map.hpp>

namespace boost {
namespace sequence {

template <class T, std::size_t N>
struct accessor<T(&)[N]>
{
    typedef index_property_map<T(&)[N]> type;
};

template <class T, std::size_t N>
struct accessor<T[N]>
{
    typedef index_property_map<T(&)[N]> type;
};

namespace array
{ 
  template <class T, std::size_t N>
  index_property_map<T(&)[N]>
  elements( T(&x)[N] )
  {
      return index_property_map<T(&)[N]>(x);
  }
} // namespace array

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_ARRAY_ELEMENTS_DWA200559_HPP
