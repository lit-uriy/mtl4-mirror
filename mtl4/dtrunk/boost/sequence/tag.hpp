// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CATEGORY_DWA2006513_HPP
# define BOOST_SEQUENCE_CATEGORY_DWA2006513_HPP

namespace boost { namespace sequence { 

namespace impl
{
  template <class S>
  struct tag;

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
