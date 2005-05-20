// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef DEREFERENCED_DWA200541_HPP
# define DEREFERENCED_DWA200541_HPP

# include <boost/sequence/detail/unspecialized.hpp>
# include <boost/iterator/iterator_traits.hpp>

namespace boost {
namespace sequence { 

// A metafunction that returns the type of cursor when dereferenced
template <class Cursor>
struct dereferenced;

namespace dereferenced_
{
  // Default implementation assumes the cursor to be an iterator
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // next(S).
  template <class Cursor, class Enable = void>
  struct implementation : detail::unspecialized, iterator_value<Cursor>
  {};
}

template <class Cursor>
struct dereferenced
  : dereferenced_::implementation<Cursor>
{};

}} // namespace boost::sequence

#endif // DEREFERENCED_DWA200541_HPP
