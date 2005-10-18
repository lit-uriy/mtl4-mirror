// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_FIXED_SIZE_INTRINSICS_DWA2005616_HPP
# define BOOST_SEQUENCE_FIXED_SIZE_INTRINSICS_DWA2005616_HPP

# include <boost/sequence/index_property_map.hpp>
# include <boost/sequence/intrinsics_fwd.hpp>
# include <boost/sequence/fixed_size/tag.hpp>
# include <boost/sequence/fixed_size/cursor.hpp>
# include <boost/sequence/range.hpp>
# include <boost/range/result_iterator.hpp>
# include <boost/array.hpp>
# include <cstddef>

namespace boost { namespace sequence { 

// Provide the implementation of intrinsics for fixed-size sequence
// types.
template <class Sequence, std::size_t N>
struct intrinsics<Sequence, fixed_size::tag<N> >
{
    struct begin
    {
        typedef fixed_size::cursor<0> type;
        type operator()(Sequence const& s) const
        { return type(); }
    };
    
    struct end
    {
        typedef fixed_size::cursor<N> type;
        type operator()(Sequence const& s) const
        { return type(); }
    };

    struct elements
    {
        typedef index_property_map<Sequence&> type;
        type operator()(Sequence& s) const
        { return type(s); }
    };
};

namespace intrinsic
{
  template <class Sequence>
  struct homogenize_fixed_size
  {
      typedef typename range_result_iterator<Sequence>::type iterator;
      typedef range<identity_property_map, iterator, iterator> type;
      
      type operator()(Sequence& a) const
      {
          return type(identity_property_map(), a.begin(), a.end());
      }
  };

  template <class T, std::size_t N>
  struct homogenize<T[N]>
    : homogenize_fixed_size<T[N]>
  {};
  
  template <class T, std::size_t N>
  struct homogenize<T const[N]>
    : homogenize_fixed_size<T const[N]>
  {};

  template <class T, std::size_t N>
  struct homogenize<array<T,N> >
    : homogenize_fixed_size<array<T,N> >
  {};
  
  template <class T, std::size_t N>
  struct homogenize<array<T,N> const>
    : homogenize_fixed_size<array<T,N> const>
  {};
}

}} // namespace boost::sequence::fixed_size

#endif // BOOST_SEQUENCE_FIXED_SIZE_INTRINSICS_DWA2005616_HPP
