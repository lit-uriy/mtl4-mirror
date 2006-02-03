// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_CONTAINER_DWA200541_HPP
# define IS_CONTAINER_DWA200541_HPP

# include <boost/sequence/core/detail/has_iterator.hpp>
# include <boost/sequence/core/detail/has_const_iterator.hpp>
# include <boost/sequence/core/detail/has_value_type.hpp>
# include <boost/mpl/and.hpp>

namespace boost { namespace sequence {
namespace detail {

// Discriminator for STL containers.  We could use stricter
// conditions, but this should do.
template <class T>
struct is_container
  : mpl::and_<
      , detail::has_iterator<T>
      , detail::has_const_iterator<T>
      , detail::has_value_type<T>
    >
{};

}}} // namespace boost::sequence::detail

#endif // IS_CONTAINER_DWA200541_HPP
