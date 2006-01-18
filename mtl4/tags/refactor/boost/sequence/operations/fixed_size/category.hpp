// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
# define BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP

# include <boost/sequence/operations/category_fwd.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence { 

namespace fixed_size
{
  // Any fixed-size sequence
  struct category : o1_size_tag
  {
      typedef category type; // self-returning, for convenience
  };
}

template <class T, std::size_t N>
struct category<T[N]>
  : fixed_size::category {};

template <class T, std::size_t N>
struct category<boost::array<T,N> >
  : fixed_size::category {};

}} // namespace boost::sequence

#endif // BOOST_FIXED_SIZE_SEQUENCE_CATEGORY_DWA200559_HPP
