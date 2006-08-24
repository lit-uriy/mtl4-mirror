// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP
# define BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP

# include <boost/sequence/project_elements.hpp>
# include <boost/sequence/lower_bound.hpp>
# include <boost/sequence/compose.hpp>
# include <boost/detail/transfer_cv.hpp>
# include <boost/functional/project1st.hpp>
# include <boost/functional/project2nd.hpp>
# include <boost/compressed_pair.hpp>
# include <boost/detail/compressed_tuple.hpp>
# include <boost/detail/compressed_single.hpp>
# include <boost/detail/callable.hpp>
# include <boost/type_traits/remove_reference.hpp>
# include <boost/indexed_storage/concepts.hpp>
# include <boost/indexed_storage/indices.hpp>
# include <boost/utility/result_of.hpp>

# if 0
#  include <boost/spirit/phoenix/bind.hpp>
# endif 
# include <boost/functional/less.hpp>
# include <boost/mpl/vector.hpp>

namespace boost { namespace indexed_storage {

namespace sparse_
{
  template  <class Cmp = functional::less>
  struct sorted
    : private boost::detail::compressed_single<Cmp>
  {
      sorted(Cmp const& cmp = Cmp())
        : boost::detail::compressed_single<Cmp>(cmp)
      {}

      template <class Storage, class SeekIndex>
      typename concepts::Sequence<Storage>::cursor
      operator()(Storage& s, SeekIndex const& i) const
      {
          BOOST_CONCEPT_ASSERT((concepts::Sequence<Storage>));
          
          typedef sequence::project_elements<Storage, functional::project1st> create_projection;
          typedef typename create_projection::result_type access_indices;
          
          typedef typename concepts::Sequence<Storage>::cursor cursor;
          
          typedef typename result_of<
              sequence::op::compose(cursor, cursor, access_indices)
          >::type index_sequence;
          BOOST_CONCEPT_ASSERT((concepts::Sequence<index_sequence>));

          typedef typename concepts::Sequence<index_sequence>::value_type index;
          
          BOOST_CONCEPT_ASSERT((
               BinaryPredicate<Cmp, index, SeekIndex>
          ));
               
          return sequence::lower_bound(
              sequence::compose(sequence::begin(s), sequence::end(s), create_projection()(s))
            , i, this->first() );
      }
  };

  namespace compressed_tuple = boost::detail::compressed_tuple;

  struct get_default
  {
      template <class Signature> struct result;

      template <class This, class S, class C, class I>
      struct result<This(S,C,I)>
      {
          typedef typename remove_reference<
              typename add_const<S>::type
          >::type seq;

          typedef typename concepts::Sequence<
              seq
          >::value_type::second_type type;
      };

      template <class S, class C, class I>
      typename result_of<get_default(S,C,I)>::type
      operator()(S const&, C const&, I const&) const
      {
          return typename concepts::Sequence<S>::value_type::first_type();
      }
  };

  struct insert_sorted : boost::detail::callable<insert_sorted>
  {
      typedef void result_type;

      template <class S, class C, class I, class V>
      void call(S& s, C& c, I& i, V& v) const
      {
          sequence::insert(s, c, std::make_pair(i,v));
      }
  };

  using boost::detail::transfer_cv;

  namespace impl
  {
    template <class N, class S>
    struct get_member_
    {
        typedef typename result_of<
            compressed_tuple::op::get(
                N&, typename transfer_cv<S,typename S::members_type>::type&
            )
        >::type result_type;

        result_type operator()(S& s) const
        {
            return compressed_tuple::get(N(), s.members);
        }
    };
  }

  namespace op
  {
    using mpl::_;
    using boost::detail::function1;
    using mpl::int_;

    struct lookup : function1<impl::get_member_<int_<0>,_> > {};
    struct get_unstored : function1<impl::get_member_<int_<1>,_> > {};
    struct set_unstored : function1<impl::get_member_<int_<2>,_> > {};
    struct storage : function1<impl::get_member_<int_<3>,_> > {};
  }

  namespace
  {
    op::lookup const& lookup = boost::detail::pod_singleton<op::lookup>::instance;
    op::get_unstored const& get_unstored = boost::detail::pod_singleton<op::get_unstored>::instance;
    op::set_unstored const& set_unstored = boost::detail::pod_singleton<op::set_unstored>::instance;
    op::storage const& storage = boost::detail::pod_singleton<op::storage>::instance;
  }

  template <
      class PairSequence
    , class Lookup = sorted<>
    , class GetUnstored = get_default
    , class SetUnstored = insert_sorted
  >
  struct sparse
  {
      typedef Lookup lookup;
      typedef GetUnstored get_unstored;
      typedef SetUnstored set_unstored;
      typedef PairSequence storage;
      typedef typename concepts::Sequence<storage>::value_type pair_type;
      typedef typename pair_type::first_type index_type;
      typedef typename pair_type::second_type value_type;

      sparse() {}

   private:
      template <class N, class S> friend struct impl::get_member_;

      typedef typename boost::detail::compressed_tuple::generate<
          mpl::vector<Lookup,GetUnstored,SetUnstored,PairSequence>
      >::type members_type;

      members_type members;
  };

  struct tag {};

} // namespace sparse_

using sparse_::sparse;
  
namespace impl
{
  template <class S>
  struct indices<S, indexed_storage::sparse_::tag>
  {
      typedef sequence::project_elements<
          typename boost::detail::transfer_cv<S,typename S::storage>::type
        , functional::project1st
      > impl;
      
      typedef typename impl::result_type result_type;
      
      result_type operator()(S& s) const
      {
          return impl()( sparse_::storage(s) );
      }
  };

  template <class S, class I>
  struct get_at<S, I, indexed_storage::sparse_::tag>
  {

      typedef typename
        concepts::Sequence<typename S::storage>::value_type::second_type
      result_type;

      result_type operator()(S& s, I& i)
      {
          typename concepts::Sequence<S>::cursor
              pos = sparse_::lookup(s)( sparse_::storage(s), i );

          if (pos != sequence::end(s))
          {
              typename boost::detail::transfer_cv<
                  typename boost::detail::transfer_cv<S, typename S::storage>::type
                , typename concepts::Sequence<typename S::storage>::value_type
              >::type x = sequence::elements(sparse_::storage(s))( *pos );

              if (x.first == i)
                  return x.second;
          }
          return sparse_::get_unstored(s)( sparse_::storage(s), pos, i );
      }
  };

  template <class S, class I, class V>
  struct set_at<S, I, V, indexed_storage::sparse_::tag>
  {
      BOOST_CONCEPT_ASSERT(
          (sequence::concepts::InsertableSequence<typename S::storage>)
      );

      typedef void result_type;

      result_type operator()(S& s, I& i, V& v)
      {
          typename concepts::Sequence<S>::cursor
              pos = sparse_::lookup(s)( sparse_::storage(s),i );

          if (pos != sequence::end(s))
          {
              typename concepts::Sequence<typename S::storage>::reference
                  x = sequence::elements( sparse_::storage(s) )(*pos);

              if (x.first == i)
                  x.second = v;
          }
          sparse_::set_unstored(s)( sparse_::storage(s), pos, i, v );
      }
  };
}
}

namespace sequence { namespace impl
{
  template <class PairSequence, class Lookup, class GetUnstored, class SetUnstored>
  struct tag<
      indexed_storage::sparse<PairSequence, Lookup, GetUnstored, SetUnstored>
  >
  {
      typedef indexed_storage::sparse_::tag type;
  };

  template <class S>
  struct begin<S, indexed_storage::sparse_::tag>
  {
      typedef typename
        boost::detail::transfer_cv<S, typename S::storage>::type
      storage;

      typedef typename begin<storage>::result_type result_type;
      result_type operator()(S& s)
      {
          return sequence::begin( indexed_storage::sparse_::storage(s) );
      }
  };
      
  template <class S>
  struct end<S, indexed_storage::sparse_::tag>
  {
      typedef typename
        boost::detail::transfer_cv<S, typename S::storage>::type
      storage;

      typedef typename end<storage>::result_type result_type;
      result_type operator()(S& s)
      {
          return sequence::end( indexed_storage::sparse_::storage(s) );
      }
  };
      
  template <class S>
  struct elements<S, indexed_storage::sparse_::tag>
  {
      typedef sequence::project_elements<
          typename boost::detail::transfer_cv<S,typename S::storage>::type
        , functional::project2nd
      > impl;

      typedef typename impl::result_type result_type;

      result_type operator()( S& s ) const
      {
          return impl()( indexed_storage::sparse_::storage(s) );
      }
  };
}} // namespace sequence::impl

} // namespace boost

#endif // BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP
