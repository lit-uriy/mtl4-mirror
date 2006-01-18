// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_DETAIL_INTRINSIC_ARG_DWA2005812_HPP
# define BOOST_SEQUENCE_DETAIL_INTRINSIC_ARG_DWA2005812_HPP

# include <boost/type_traits/remove_const.hpp>
# include <boost/type_traits/remove_reference.hpp>

namespace boost { namespace sequence { namespace detail { 

// Converts actual argument types for result<...> (used by
// boost::result_of) into the argument type to a class template that
// implements an intrinsic function.
template <class T>
struct intrinsic_arg  // Rvalues are treated as const
{
    typedef T const type; 
};

template <class T>
struct intrinsic_arg<T const&>  // Const lvalues are treated as const
{
    typedef T const type;
};

template <class T>
struct intrinsic_arg<T&> // Non-const lvalues are treated as non-const
{
    typedef T type;
};

// Converts actual argument types for result<...> (used by
// boost::result_of) into the argument type to a class template that
// implements an intrinsic function whose argument is always const
template <class T>
struct intrinsic_const_arg
  : boost::remove_const<
        typename boost::remove_reference<T>::type
    >
{};

}}} // namespace boost::sequence::detail

#endif // BOOST_SEQUENCE_DETAIL_INTRINSIC_ARG_DWA2005812_HPP
