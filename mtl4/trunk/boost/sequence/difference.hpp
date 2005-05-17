// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef DIFFERENCE_DWA200541_HPP
# define DIFFERENCE_DWA200541_HPP

# include <boost/sequence/detail/unspecialized.hpp>
# include <cstddef>

namespace boost {
namespace sequence { 

// A metafunction that returns the type of the difference between the
// two given cursors
template <class Cursor1, class Cursor2>
struct difference;

namespace difference_
{
  // Default implementation assumes the cursor homogneous
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // next(S).
  template <class Cursor1, class Cursor2, class Enable = void>
  struct implementation : detail::unspecialized
  {
      typedef std::ptrdiff_t type;
  };
}

template <class Cursor1, class Cursor2>
struct difference
  : difference_::implementation<Cursor1, Cursor2>
{};

}} // namespace boost::sequence

#endif // DIFFERENCE_DWA200541_HPP
