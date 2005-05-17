// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef HAS_VALUE_DWA2005331_HPP
# define HAS_VALUE_DWA2005331_HPP

# include <boost/mpl/bool.hpp>

namespace boost {
namespace sequence {
namespace detail {

template <class T, class U = int[1]>
struct has_value
  : mpl::false_ {};
  
template <class T>
struct has_value<T, int[T::value * 0 + 1]>
  : mpl::true_ {};

}}} // namespace boost::sequence::detail

#endif // HAS_VALUE_DWA2005331_HPP
