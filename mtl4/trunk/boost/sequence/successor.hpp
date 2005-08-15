#error obsolete
// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef SUCCESSOR_DWA200541_HPP
# define SUCCESSOR_DWA200541_HPP

# include <boost/sequence/detail/unspecialized.hpp>


namespace boost {
namespace sequence { 

// A metafunction that returns the type of a cursor's successor
template <class Cursor>
struct successor;

namespace successor_
{
  // Default implementation assumes the cursor homogneous
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // next(S).
  template <class Cursor, class Enable = void>
  struct implementation : detail::unspecialized
  {
      typedef Cursor type;
  };
}

template <class Cursor>
struct successor
  : successor_::implementation<Cursor>
{};

template <class Cursor>
struct successor<Cursor const>
  : successor<Cursor>
{};

}} // namespace boost::sequence

#endif // SUCCESSOR_DWA200541_HPP
