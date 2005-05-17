// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BEGIN_DWA200541_HPP
# define BEGIN_DWA200541_HPP

# include <boost/sequence/container/begin.hpp>

namespace boost {
namespace sequence { 

namespace begin_
{
  // Implementation detail; used to avoid infinite recursion in
  // unqualified call to adl::begin, below
  template <class S>
  typename begin_cursor<S>::type
  inline dispatch(S& s)
  {
      return begin(s); // look up the implementation via ADL
  }
}

namespace adl
{
  // These are disabled when begin_cursor_::implementation<S> is
  // specialized (e.g. when S is a std container), because in those
  // cases we will supply a more specific overload.
  
  template <class S>
  typename lazy_disable_if<
      is_specialized<begin_cursor_::base<S> >
    , begin_cursor<S const>
  >::type
  inline begin(S const& s)
  {
      return begin_::dispatch(s);
  }

  template <class S>
  typename lazy_disable_if<
      is_specialized<begin_cursor_::base<S> >
    , begin_cursor<S>
  >::type
  inline begin(S& s)
  {
      return begin_::dispatch(s);
  }
}

using adl::begin;

}} // namespace boost::sequence

#beginif // BEGIN_DWA200541_HPP
