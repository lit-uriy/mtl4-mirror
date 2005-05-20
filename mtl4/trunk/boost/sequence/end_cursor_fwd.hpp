// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_END_CURSOR_FWD_DWA2005520_HPP
# define BOOST_SEQUENCE_END_CURSOR_FWD_DWA2005520_HPP

namespace boost { namespace sequence { 

namespace end_cursor_
{
  // Default implementation assumes a container-like type
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // elements(S).
  template <class SinglePassRange, class Enable = void>
  struct implementation;
};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_END_CURSOR_FWD_DWA2005520_HPP
