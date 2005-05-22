// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ACCESSOR_DWA200541_HPP
# define ACCESSOR_DWA200541_HPP

# include <boost/sequence/accessor_fwd.hpp>
# include <boost/sequence/identity_property_map.hpp>
# include <boost/sequence/fixed_size/accessor.hpp>
# include <boost/sequence/detail/unspecialized.hpp>

namespace boost {
namespace sequence { 

// A metafunction that returns the type of the property map associated
// with a sequence.
template <class Sequence>
struct accessor;

namespace accessor_
{
  // Default implementation assumes a container-like type
  
  // To support known types whose namespaces we can't enter, we add an
  // Enable parameter to use with SFINAE.  Partially specialize and
  // enable for specific types.  Partial specializations will be
  // detected so you can use enable_if to create overloads of
  // elements(S).
  template <class SinglePassRange, class Enable>
  struct implementation : detail::unspecialized
  {
      typedef identity_property_map type;
  };
}

template <class Sequence>
struct accessor
  : accessor_::implementation<Sequence>
{};

}} // namespace boost::sequence

#endif // ACCESSOR_DWA200541_HPP
