// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef ELEMENTS_DWA200541_HPP
# define ELEMENTS_DWA200541_HPP

# include <boost/sequence/container/elements.hpp>
# include <boost/sequence/fixed_size/elements.hpp>
# include <boost/sequence/accessor.hpp>
# include <boost/sequence/detail/is_specialized.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost {
namespace sequence { 

namespace elements_
{
  // Implementation detail; used to avoid infinite recursion in
  // unqualified call to adl::elements, below
  template <class S>
  typename accessor<S>::type
  inline dispatch(S& s)
  {
      return elements(s); // look up the implementation via ADL
  }
}

# if !BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
namespace adl
{
# endif 
  // These are disabled when accessor_::implementation<S> is
  // specialized (e.g. when S is a std container), because in those
  // cases we will supply a more specific overload.
  
  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<accessor_::implementation<S,void> >
    , accessor<S const>
  >::type
  inline elements(S const& s)
  {
      return elements_::dispatch(s);
  }

  template <class S>
  typename lazy_disable_if<
      detail::is_specialized<accessor_::implementation<S,void> >
    , accessor<S>
  >::type
  inline elements(S& s)
  {
      return elements_::dispatch(s);
  }
# if !BOOST_WORKAROUND(__GNUC__, BOOST_TESTED_AT(4))
}
using adl::elements;
# endif 

}} // namespace boost::sequence

#endif // ELEMENTS_DWA200541_HPP
