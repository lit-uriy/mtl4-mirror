// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BEGIN_DWA200541_HPP
# define BEGIN_DWA200541_HPP

# include <boost/sequence/fixed_size/begin.hpp>
# include <boost/sequence/detail/is_specialized.hpp>

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

# if !BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
namespace adl
{
# endif 
  // These are disabled when begin_cursor_::implementation<S> is
  // specialized (e.g. when S is a std container), because in those
  // cases we will supply a more specific overload.
  
  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<begin_cursor_::implementation<S> >
    , begin_cursor<S const>
  >::type
  inline begin(S const& s)
  {
      return begin_::dispatch(s);
  }

  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<begin_cursor_::implementation<S> >
    , begin_cursor<S>
  >::type
  inline begin(S& s)
  {
      return begin_::dispatch(s);
  }
# if !BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
}
using adl::begin;
# endif 


}} // namespace boost::sequence

#endif // BEGIN_DWA200541_HPP
