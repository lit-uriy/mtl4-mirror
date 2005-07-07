// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
# define BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP

# include <boost/sequence/category_fwd.hpp>
# include <boost/sequence/algorithm/fixed_size/category.hpp>
# include <boost/sequence/range_fwd.hpp>
# include <boost/sequence/fixed_size/cursor_fwd.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence { 

template <class T, std::size_t N>
struct category<T[N]>
  : algorithm::fixed_size::homogeneous {};

template <class T, std::size_t N>
struct category<T const[N]>
  : algorithm::fixed_size::homogeneous {};

template <class T, std::size_t N>
struct category<boost::array<T,N> >
  : algorithm::fixed_size::homogeneous {};

template <class Elements, std::size_t N1, std::size_t N2>
struct category<range<Elements, fixed_size::cursor<N1>, fixed_size::cursor<N2> > >
  : algorithm::fixed_size::homogeneous {};

}} // namespace boost::sequence

#endif // BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
