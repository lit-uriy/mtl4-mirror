// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CATEGORY_DWA2006513_HPP
# define BOOST_SEQUENCE_CATEGORY_DWA2006513_HPP

# include <boost/fusion/support/is_sequence.hpp>
# include <boost/fusion/sequence/adapted/array.hpp>
# include <utility>

namespace boost { namespace sequence { 

namespace impl
{
  struct fusion_tag {};
  
  template <class S>
  struct tag
    : mpl::if_<fusion::is_sequence<S>, fusion_tag, void>
  {};

  template <class T, class U>
  struct tag<std::pair<T,U> >
  {
      typedef void type;
  };
      
  template <class S>
  struct tag<S const>
    : tag<S>
  {};
  
  template <class S>
  struct tag<S volatile>
    : tag<S>
  {};
  
  template <class S>
  struct tag<S const volatile>
    : tag<S>
  {};
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_CATEGORY_DWA2006513_HPP
