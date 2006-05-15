// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP
# define BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP

# include <boost/indexed_storage/difference1.hpp>
# include <boost/indexed_storage/address_difference1.hpp>
# include <boost/detail/is_xxx.hpp>
# include <boost/iterator/counting_iterator.hpp>
# include <boost/iterator/iterator_traits.hpp>

# include <boost/sequence/concepts.hpp>
# include <boost/detail/function1.hpp>
# include <boost/detail/pod_singleton.hpp>
# include <boost/utility/enable_if.hpp>
# include <boost/mpl/and.hpp>
# include <boost/type_traits/is_convertible.hpp>
# include <boost/type_traits/is_pointer.hpp>

# include <boost/mpl/placeholders.hpp>

namespace boost {

namespace indexed_storage { 

namespace detail
{
  BOOST_DETAIL_IS_XXX_DEF(counting_iterator,boost::counting_iterator,3)
}

namespace impl
{
  template <class S, class enable = void>
  struct indices_base;
  
  template <class S>
  struct indices
    : indices_base<S>
  {};

  // By default, every Boost.Range is a sequence whose cursors are
  // counting_iterators, so we'll provide a default implementation of
  // indices that works for that case.
  template <class S>
  struct indices_base<
      S
    , typename enable_if<
          mpl::and_<
              detail::is_counting_iterator<
                  typename sequence::concepts::Sequence<S>::cursor
              >
            , is_convertible<
                  typename iterator_traversal<
                      typename sequence::concepts::Sequence<S>::cursor
                  >::type
                , random_access_traversal_tag
              >
          >
      >::type
  >
  {
      typedef difference1<
          typename sequence::concepts::Sequence<S>::cursor
      > result_type;
      
      result_type operator()(S& s) const
      {
          return result_type(sequence::begin(s));
      }
  };

  // Sequences whose cursors are pointers can also have a default
  // indices implementation
  template <class S>
  struct indices_base<
      S
    , typename enable_if<
          is_pointer<
              typename sequence::concepts::Sequence<S>::cursor
          >
      >::type
  >
  {
      typedef address_difference1<
          typename sequence::concepts::Sequence<S>::cursor
      > result_type;
      
      result_type operator()(S& s) const
      {
          return result_type(sequence::begin(s));
      }
  };
}

namespace op
{
  struct indices : boost::detail::function1<impl::indices<mpl::_> > {};
}

namespace
{
  op::indices const& indices = boost::detail::pod_singleton<op::indices>::instance;
}

}} // namespace boost::indexed_storage

#endif // BOOST_INDEXED_STORAGE_INDICES_DWA200658_HPP
