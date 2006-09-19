// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP
# define BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP

# include <boost/detail/function2.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/tag.hpp>
# include <boost/sequence/concepts.hpp>
# include <boost/sequence/composed.hpp>
# include <boost/mpl/placeholders.hpp>

namespace boost { namespace indexed_storage { 

namespace concepts
{
  using namespace sequence::concepts;
}

namespace impl
{
  template <class S, class I, class = typename sequence::impl::tag<S>::type>
  struct get_at
  {
      typedef typename concepts::Sequence<S>::reference result_type;
      result_type operator()(S& s, I& i)
      {
          return s[i];
      }
  };

  template <class S, class I>
  struct get_at< S, I, sequence::impl::composed_tag >
  {
      typedef typename concepts::Sequence<S>::reference result_type;
      result_type operator()(S& s, I& i)
      {
          return sequence::elements(s)(*(sequence::begin(s) + i));
      }
  };
}

namespace op
{
  using mpl::_;
  struct get_at : boost::detail::function2<impl::get_at<_,_,sequence::impl::tag<_> > > {};
}

namespace
{
  op::get_at const& get_at = boost::detail::pod_singleton<op::get_at>::instance;
}

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP
