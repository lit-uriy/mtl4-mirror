// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_COMPOSE_DWA200655_HPP
# define BOOST_SEQUENCE_COMPOSE_DWA200655_HPP

# include <boost/detail/function3.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/composed.hpp>
# include <boost/mpl/placeholders.hpp>
# include <boost/mpl/assert.hpp>
# include <boost/type_traits/is_same.hpp>

namespace boost { namespace sequence { 

namespace impl
{
  template <class C0, class C1, class F>
  struct compose
  {
      BOOST_MPL_ASSERT((is_same<typename remove_cv<C0>::type, typename remove_cv<C1>::type>));
      
      typedef composed<typename remove_cv<C0>::type, F> result_type;
      
      result_type operator()(C0& start, C1& finish, F& accessor) const
      {
          return result_type(start,finish,accessor);
      }
  };
}

namespace op
{
  using mpl::_;
  struct compose : boost::detail::function3<impl::compose<_,_,_> > {};
}

namespace
{
  op::compose const& compose = boost::detail::pod_singleton<op::compose>::instance;
}

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_COMPOSE_DWA200655_HPP
