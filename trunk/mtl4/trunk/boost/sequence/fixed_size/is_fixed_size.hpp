// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_FIXED_SIZE_DWA200555_HPP
# define IS_FIXED_SIZE_DWA200555_HPP

# include <boost/array.hpp>
# include <boost/mpl/bool.hpp>
# include <cstddef>

namespace boost { namespace sequence { namespace fixed_size {

template <class T>
struct is_fixed_size
  : mpl::false_ {};

template <class T, std::size_t N>
struct is_fixed_size<T[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T(&)[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T const[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<T const(&)[N]>
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<boost::array<T,N> >
  : mpl::true_ {};

template <class T, std::size_t N>
struct is_fixed_size<boost::array<T,N> const>
  : mpl::true_ {};

}}} // namespace boost::sequence::fixed_size

#endif // IS_FIXED_SIZE_DWA200555_HPP
