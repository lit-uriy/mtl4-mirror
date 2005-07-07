// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef HAS_VALUE_DWA2005331_HPP
# define HAS_VALUE_DWA2005331_HPP

# include <boost/mpl/bool.hpp>

namespace boost {
namespace sequence {
namespace detail {

# if BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050601)) || BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(3))

typedef char (&no_tag)[1];
typedef char (&yes_tag)[2];

template <class T>
struct has_value_helper
{
	typedef int type;
	static const bool value = false;
};

template <class T>
char has_value_tester( typename has_value_helper<int[T::value*0 + 1] >::type ) ;

template <class T>
char (& has_value_tester(...) )[2];

template <class T>
struct has_value
  : mpl::bool_< (sizeof(has_value_tester<T>(0)) == 1) >
{};


# else 
template <class T, class U = int[1]>
struct has_value
  : mpl::false_ {};
  
template <class T>
struct has_value<T, int[T::value*0+1]>
  : mpl::true_ {};
# endif 

}}} // namespace boost::sequence::detail

#endif // HAS_VALUE_DWA2005331_HPP
