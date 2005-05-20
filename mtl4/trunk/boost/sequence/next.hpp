// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef NEXT_DWA200541_HPP
# define NEXT_DWA200541_HPP

# include <boost/sequence/successor.hpp>
# include <boost/sequence/detail/is_specialized.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost {
namespace sequence { 

namespace next_
{
  // Implementation detail; used to avoid infinite recursion in
  // unqualified call to adl::next, below
  template <class S>
  typename successor<S>::type
  inline dispatch(S& s)
  {
      return next(s); // look up the implementation via ADL
  }
}

namespace adl
{
  // These are disabled when successor_::implementation<S> is
  // specialized (e.g. when S is a std container), because in those
  // cases we will supply a more specific overload.
  
  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<successor_::implementation<S> >
    , successor<S const>
  >::type
  inline next(S const& s)
  {
      return next_::dispatch(s);
  }

  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<successor_::implementation<S> >
    , successor<S>
  >::type
  inline next(S& s)
  {
      return next_::dispatch(s);
  }
}

using adl::next;

}} // namespace boost::sequence

#endif // NEXT_DWA200541_HPP
