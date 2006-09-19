// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_CURSOR_PRIOR_DWA2006919_HPP
# define BOOST_CURSOR_PRIOR_DWA2006919_HPP

# include <boost/detail/function1_by_value.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/fusion/iterator/prior.hpp>
# include <boost/cursor/tag.hpp>
# include <boost/detail/remove_cv_reference.hpp>
# include <boost/mpl/placeholders.hpp>

namespace boost { namespace cursor { 

namespace impl
{
  template <class C, class = typename tag<C>::type>
  struct prior;

  template <class C>
  struct prior<C,homogeneous_tag>
  {
      typedef C result_type;
      
      result_type operator()(C p) const
      {
          return --p;
      }
  };

  template <class C>
  struct prior<C,heterogeneous_tag>
  {
      typedef typename fusion::result_of::prior<C>::type result_type;
      
      result_type operator()(C p) const
      {
          return fusion::prior(p);
      }
  };
}

namespace op
{
  using mpl::_;
  struct prior : boost::detail::function1_by_value<impl::prior<_,impl::tag<_> > > {};
}

namespace
{
  op::prior const& prior = boost::detail::pod_singleton<op::prior>::instance;
}

}} // namespace boost::cursor

#endif // BOOST_CURSOR_PRIOR_DWA2006919_HPP
