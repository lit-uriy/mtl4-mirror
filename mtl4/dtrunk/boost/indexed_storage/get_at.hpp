// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP
# define BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP

# include <boost/detail/function2.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/sequence/concepts.hpp>
# include <boost/sequence/composed.hpp>

namespace boost { namespace indexed_storage { 

namespace concepts
{
  using namespace sequence::concepts;
}

namespace impl
{
  template <class S, class I>
  struct get_at
  {
      typedef typename concepts::Sequence<S>::reference result_type;
      result_type operator()(S& s, I& i)
      {
          return s[i];
      }
  };

  template <class Iter, class Mapping, class Index>
  struct get_at< sequence::composed<Iter,Mapping> const, Index >
  {
      typedef sequence::composed<Iter,Mapping> const S;
      
      typedef typename concepts::Sequence<S>::reference result_type;
      result_type operator()(S& s, Index& i)
      {
          return sequence::elements(s)(*(sequence::begin(s) + i));
      }
  };

  template <class Iter, class Mapping, class Index>
  struct get_at< sequence::composed<Iter,Mapping>, Index >
    : get_at< sequence::composed<Iter,Mapping> const, Index >
  {};
}

namespace op
{
  struct get_at : boost::detail::function2<impl::get_at> {};
}

namespace
{
  op::get_at const& get_at = boost::detail::pod_singleton<op::get_at>::instance;
}

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_GET_AT_DWA2006514_HPP
