// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IS_MPL_INTEGRAL_CONSTANT_DWA200541_HPP
# define IS_MPL_INTEGRAL_CONSTANT_DWA200541_HPP

# include <boost/sequence/core/detail/has_value_type.hpp>
# include <boost/sequence/core/detail/has_type.hpp>
# include <boost/sequence/core/detail/has_value.hpp>
# include <boost/type_traits/is_same.hpp>
# include <boost/mpl/and.hpp>
# include <boost/mpl/bool.hpp>

namespace boost { namespace sequence { namespace detail {
        
template <class T>
struct is_self_returning_nullary_metafunction
  : is_same<typename T::type,T>
{};

template <class T>
struct is_mpl_integral_constant
  : mpl::and_<
        detail::has_type<T>
      , detail::is_self_returning_nullary_metafunction<T>
      , detail::has_value_type<T>
      , detail::has_value<T>
    >
{};

}}} // namespace boost::sequence::detail

#endif // IS_MPL_INTEGRAL_CONSTANT_DWA200541_HPP
