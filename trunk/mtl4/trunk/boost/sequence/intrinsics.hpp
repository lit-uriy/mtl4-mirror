// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP

# include <boost/sequence/intrinsics_fwd.hpp>
# include <boost/sequence/fixed_size/intrinsics.hpp>
# include <boost/sequence/fixed_size/tag.hpp>
# include <boost/sequence/identity_property_map.hpp>
# include <boost/sequence/iterator_range_tag.hpp>

# include <boost/range/begin.hpp>
# include <boost/range/end.hpp>
# include <boost/range/iterator.hpp>
# include <boost/range/const_iterator.hpp>

# include <boost/iterator/iterator_traits.hpp>

# include <boost/type_traits/is_array.hpp>
# include <boost/utility/enable_if.hpp>

namespace boost { namespace sequence {

// The implementation of the intrinsics specialization that applies to
// models of SinglePassRange (see the Boost Range library
// documentation at http://www.boost.org/libs/range/doc/range.html).
// The first argument is the range type and the second argument is a
// nullary metafunction that returns the iterator type to use as the
// Range's cursor.
//
// Note: the need for the 2nd argument may disappear when the Range
// library is changed so that range_iterator<R const>::type is the
// same as range_const_iterator<R>::type.
template <class Sequence, class GetIterator>
struct iterator_range_intrinsics
{
    struct begin
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::begin(s);
        }
    };
        
    struct end
    {
        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return boost::end(s);
        }
    };
        
    struct elements
    {

        typedef typename GetIterator::type type;
        
        type operator()(Sequence& s) const
        {
            return type();
        }
    };
};

// Intrinsics specializations for iterator ranges.
template <class Sequence>
struct intrinsics<Sequence, iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_iterator<Sequence>
    >
{};

template <class Sequence>
struct intrinsics<Sequence const, iterator_range_tag>
  : iterator_range_intrinsics<
        Sequence, range_const_iterator<Sequence>
    >
{};

namespace intrinsic
{
  // The default implementation of each intrinsic function object type
  // is inherited from the corresponding member of
  // intrinsics<Sequence>.  You can of course specialize begin<S>,
  // end<S>, and elements<S>, individually, but specializing
  // intrinsics<> usually more convenient.
  template <class Sequence>
  struct begin : intrinsics<Sequence>::begin {};
  
  template <class Sequence>
  struct end : intrinsics<Sequence>::end {};
  
  template <class Sequence>
  struct elements : intrinsics<Sequence>::elements {};

  template <class Cursor>
  struct next
  {
      typedef Cursor type;
      Cursor operator()(Cursor x) const
      {
          return ++x;
      }
  };

  template <class Cursor>
  struct next<Cursor const>
    : next<Cursor>
  {};
  
  template <class Cursor>
  struct prev
  {
      typedef Cursor type;
      Cursor operator()(Cursor x) const
      {
          return --x;
      }
  };
  
  template <class Cursor>
  struct prev<Cursor const>
    : prev<Cursor>
  {};
  
  template <class Cursor1, class Cursor2>
  struct equal
  {
      typedef bool type;
      
      type operator()(Cursor1 const& c1, Cursor2 const& c2) const
      {
          return c2 == c1;
      }
  };

  template <class Cursor1, class Cursor2>
  struct equal<Cursor1 const, Cursor2 const>
    : equal<Cursor1,Cursor2>
  {};
  
  template <class Cursor1, class Cursor2>
  struct equal<Cursor1 const, Cursor2>
    : equal<Cursor1,Cursor2>
  {};
  
  template <class Cursor1, class Cursor2>
  struct equal<Cursor1, Cursor2 const>
    : equal<Cursor1,Cursor2>
  {};
  
  
  template <class Cursor1, class Cursor2>
  struct distance
  {
      typedef typename iterator_difference<Cursor1>::type type;
      
      type operator()(Cursor1 const& c1, Cursor2 const& c2) const
      {
          return c2 - c1;
      }
  };

  template <class Cursor1, class Cursor2>
  struct distance<Cursor1 const, Cursor2 const>
    : distance<Cursor1,Cursor2>
  {};
  
  template <class Cursor1, class Cursor2>
  struct distance<Cursor1 const, Cursor2>
    : distance<Cursor1,Cursor2>
  {};
  
  template <class Cursor1, class Cursor2>
  struct distance<Cursor1, Cursor2 const>
    : distance<Cursor1,Cursor2>
  {};
  
  template <class Cursor, class Distance>
  struct advance
  {
      typedef Cursor type;
      
      type operator()(Cursor c, Distance const& d) const
      {
          return c += d;
      }
  };
  
  template <class Cursor, class Distance>
  struct advance<Cursor const, Distance const>
    : advance<Cursor,Distance>
  {};
  
  template <class Cursor, class Distance>
  struct advance<Cursor const, Distance>
    : advance<Cursor,Distance>
  {};
  
  template <class Cursor, class Distance>
  struct advance<Cursor, Distance const>
    : advance<Cursor,Distance>
  {};
}


}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INTRINSICS_DWA2005616_HPP
