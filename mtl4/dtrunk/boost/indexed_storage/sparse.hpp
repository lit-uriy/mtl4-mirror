// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP
# define BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP

# include <boost/sequence/project_elements.hpp>
# include <boost/detail/transfer_cv.hpp>
# include <boost/detail/project1st.hpp>
# include <boost/detail/project2nd.hpp>

namespace boost {

namespace indexed_storage
{ 
  template <class PairSequence, class LookupPolicy>
  struct sparse : LookupPolicy
  {
      typedef PairSequence storage;
      typedef typename Sequence<PairSequence>::value_type pair_type;
      typedef typename pair_type::first_type index_type;
      typedef typename pair_type::second_type value_type;

      sparse() {}
      
      sparse(
          typename add_reference<
              typename add_const<storage>::type
          >::type pairs
      )
        : pairs(pairs)
      {}
      
      storage pairs;
  };

  struct sparse_tag {};

  namespace impl
  {
    template <class S>
    struct indices<S, indexed_storage::sparse_tag>
      : sequence::project_elements<S,boost::detail::project1st>
    {};

    template <class S, class I>
    struct get_at<S, I, indexed_storage::sparse_tag>
    {
        typedef typename
        detail::transfer_cv<
            typename detail::transfer_cv<S, typename S::storage>::type
          , S::value_type
        >::type& result_type;

        result_type operator()(S& s, I& i)
        {
            return sequence::elements(s)(*s.find(s.pairs,i));
        }
    };

    template <class S, class I, class V>
    struct get_at<S, I, V, indexed_storage::sparse_tag>
    {
        typedef typename
        detail::transfer_cv<
            typename detail::transfer_cv<S, typename S::storage>::type
          , S::value_type
        >::type& result_type;

        result_type operator()(S& s, I& i, V& v)
        {
            return sequence::elements(s)(*s.find(s.pairs,i));
        }
    };
  }
}

namespace sequence { namespace impl
{
  template <class PairContainer>
  struct tag<indexed_storage<PairContainer> >
  {
      typedef indexed_storage::sparse_tag type;
  };

  template <class S>
  struct begin<S, indexed_storage::sparse_tag>
  {
      typedef typename
        detail::transfer_cv<S, typename S::storage>::type
      storage;

      typedef typename begin<storage>::result_type result_type;
      result_type operator()(S& s)
      {
          return sequence::begin(s.pairs);
      }
  };
      
  template <class S>
  struct end<S, indexed_storage::sparse_tag>
  {
      typedef typename
        detail::transfer_cv<S, typename S::storage>::type
      storage;

      typedef typename end<storage>::result_type result_type;
      result_type operator()(S& s)
      {
          return sequence::end(s.pairs);
      }
  };
      
  template <class S>
  struct elements<S, indexed_storage::sparse_tag>
    : sequence::project_elements<S,boost::detail::project2nd>
  {};
}} // namespace sequence::impl

} // namespace boost

#endif // BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP
