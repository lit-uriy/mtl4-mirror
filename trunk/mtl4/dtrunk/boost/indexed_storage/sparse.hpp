// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP
# define BOOST_INDEXED_STORAGE_SPARSE_DWA2006515_HPP

# include <boost/sequence/project_elements.hpp>
# include <boost/detail/transfer_cv.hpp>
# include <boost/detail/project1st.hpp>
# include <boost/detail/project2nd.hpp>
# include <boost/detail/compressed_pair.hpp>

namespace boost {

namespace indexed_storage
{
  template <class PairSequence, class Lookup, class GetUnstored, class SetUnstored>
  struct sparse
  {
      typedef PairSequence storage;
      typedef typename concepts::Sequence<PairSequence>::value_type pair_type;
      typedef typename pair_type::first_type index_type;
      typedef typename pair_type::second_type value_type;

      sparse() {}
      
      storage& lookup() { return members.first(); }
      Lookup& get_unstored() { return members.second().first(); }
      GetUnstored& set_unstored() { return members.second().second().first(); }
      SetUnstored& pairs() { return members.second().second().second(); }
      
      storage const& lookup() const { return members.first(); }
      Lookup const& get_unstored() const { return members.second().first(); }
      GetUnstored const& set_unstored() const { return members.second().second().first(); }
      SetUnstored const& pairs() const { return members.second().second().second(); }
      
   private:
      compressed_pair<
          Lookup
        , compressed_pair<
              GetUnstored
            , compressed_pair<
                  SetUnstored
                , PairSequence
              >
          >
      > members;
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
        
        detail::transfer_cv<
            typename detail::transfer_cv<S, typename S::storage>::type
          , S::value_type
        >::type& result_type;

        result_type operator()(S& s, I& i)
        {
            typename concepts::Sequence<S>::cursor
                pos = s.lookup()(s.pairs,i);

            if (pos != sequence::end(s))
            {
                typename concepts::Sequence<typename S::storage>::reference
                    x = sequence::elements(s.pairs())(*pos);
                    
                if (x.first == i)
                    return x.second;
            }
            return s.get_unstored()(s.pairs(), pos, i);
        }
    };

    template <class S, class I, class V>
    struct set_at<S, I, V, indexed_storage::sparse_tag>
    {
        typedef void result_type;

        result_type operator()(S& s, I& i, V& v)
        {
            typename concepts::Sequence<S>::cursor
                pos = s.lookup()(s.pairs,i);

            if (pos != sequence::end(s))
            {
                typename concepts::Sequence<typename S::storage>::reference
                    x = sequence::elements(s.pairs())(*pos);
                    
                if (x.first == i)
                    x.second = v;
            }
            return s.set_unstored()(s.pairs(), pos, i, v);
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
