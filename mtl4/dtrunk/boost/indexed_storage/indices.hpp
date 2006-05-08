// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP
# define BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP

# include <boost/indexed_storage/index_map.hpp>
# include <boost/sequence/concepts.hpp>
# include <boost/iterator/iterator_traits.hpp>
# include <boost/detail/function1.hpp>
# include <boost/type_traits/is_convertible.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace indexed_storage { 

namespace impl
{
  template <class S, class enable = void>
  struct indices;

  template <
      class S
    , typename enable_if<
          is_convertible<
              typename iterator_traversal<sequence::concepts::Sequence<S>::cursor>::type
            , random_access_traversal_tag
          >
      >
  >
  struct indices
  {
      typedef index_map<sequence::concepts::Sequence<S>::cursor> result_type;
      
      result_type operator()(S& s) const
      {
          return result_type(boost::indices(s));
      }
  };
}

namespace op
{
  struct indices : boost::detail::function1<impl::indices> {};
}

namespace
{
  op::indices const& indices = boost::detail::pod_singleton<op::indices>::instance;
}

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP
