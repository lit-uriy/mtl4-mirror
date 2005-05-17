// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef END_DWA200541_HPP
# define END_DWA200541_HPP

# include <boost/sequence/container/end.hpp>

namespace boost {
namespace sequence { 

namespace end_
{
  // Implementation detail; used to avoid infinite recursion in
  // unqualified call to adl::end, below
  template <class S>
  typename end_cursor<S>::type
  inline dispatch(S& s)
  {
      return end(s); // look up the implementation via ADL
  }
}

namespace adl
{
  // These are disabled when end_cursor_::implementation<S> is
  // specialized (e.g. when S is a std container), because in those
  // cases we will supply a more specific overload.
  
  template <class S>
  typename lazy_disable_if<
      is_specialized<end_cursor_::base<S> >
    , end_cursor<S const>
  >::type
  inline end(S const& s)
  {
      return end_::dispatch(s);
  }

  template <class S>
  typename lazy_disable_if<
      is_specialized<end_cursor_::base<S> >
    , end_cursor<S>
  >::type
  inline end(S& s)
  {
      return end_::dispatch(s);
  }
}

using adl::end;

}} // namespace boost::sequence

#endif // END_DWA200541_HPP
