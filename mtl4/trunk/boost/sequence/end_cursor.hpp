// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef END_CURSOR_DWA200541_HPP
# define END_CURSOR_DWA200541_HPP

# include <boost/sequence/end_cursor_fwd.hpp>
# include <boost/sequence/detail/unspecialized.hpp>
# include <boost/sequence/fixed_size/end_cursor.hpp>
# include <boost/range/result_iterator.hpp>

namespace boost {
namespace sequence { 

// A metafunction that returns the type of the property map associated
// with a sequence.
template <class Sequence>
struct end_cursor;

namespace end_cursor_
{
  // Default implementation.
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // end(S).
  template <class SinglePassRange, class Enable>
  struct implementation
    : detail::unspecialized
    , range_result_iterator<SinglePassRange>
  {};
}

template <class Sequence>
struct end_cursor
  : end_cursor_::implementation<Sequence>
{};

}} // namespace boost::sequence

#endif // END_CURSOR_DWA200541_HPP
