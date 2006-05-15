// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_SET_AT_DWA2006514_HPP
# define BOOST_INDEXED_STORAGE_SET_AT_DWA2006514_HPP

# include <boost/detail/function3.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/mpl/placeholders.hpp>

namespace boost { namespace indexed_storage { 

namespace impl
{
  template <class S, class I, class V, class = typename sequence::impl::tag<S>::type>
  struct set_at
  {
      typedef void result_type;
      
      result_type operator()(S& s, I& i, V& v)
      {
          s[i] = v;
      }
  };
  
  template <class S, class I, class V>
  struct set_at<S,I,V,sequence::impl::composed_tag>
  {
      typedef void result_type;

      result_type operator()(S& s, I& i, V& v)
      {
          sequence::elements(s)(*(sequence::begin(s) + i), v);
      }
  };
}

namespace op
{
  using mpl::_;
  struct set_at : boost::detail::function3<impl::set_at<_,_,_,sequence::impl::tag<_> > > {};
}

namespace
{
  op::set_at const& set_at = boost::detail::pod_singleton<op::set_at>::instance;
}

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_SET_AT_DWA2006514_HPP
