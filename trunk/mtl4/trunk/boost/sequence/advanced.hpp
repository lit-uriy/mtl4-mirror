// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ADVANCED_DWA200541_HPP
# define ADVANCED_DWA200541_HPP

# include <boost/sequence/detail/unspecialized.hpp>
# include <boost/sequence/fixed_size/advanced.hpp>

namespace boost {
namespace sequence { 

// A metafunction that returns the type of cursor when advanced by the
// given Amount
template <class Cursor, class Amount>
struct advanced;

namespace advanced_
{
  // Default implementation assumes the cursor homogeneous
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // next(S).
  template <class Cursor, class Amount, class Enable = void>
  struct implementation : detail::unspecialized
  {
      typedef Cursor type;
  };
}

template <class Cursor, class Amount>
struct advanced
  : advanced_::implementation<Cursor, Amount>
{};

}} // namespace boost::sequence

#endif // ADVANCED_DWA200541_HPP
