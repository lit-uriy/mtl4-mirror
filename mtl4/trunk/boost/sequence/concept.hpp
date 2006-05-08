// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CONCEPT_DWA200652_HPP
# define BOOST_SEQUENCE_CONCEPT_DWA200652_HPP

# include <boost/property_map/concept.hpp>
# include <boost/sequence/begin.hpp>
# include <boost/sequence/end.hpp>
# include <boost/sequence/elements.hpp>

namespace boost { namespace sequence { 

template <class S>
struct Sequence
  : ReadablePropertyMap<
        typename result_of<op::elements(S)>::type
      , typename result_of<op::begin(S)>::type
    >
{
    // Associated types cursor, elements, key_type, value_type,
    // and reference, all come from ReadablePropertyMap

    // The end cursor doesn't have to have the same type as the begin
    // cursor, just as long as you can compare them.
    typedef typename result_of<op::end(S)>::type  end_cursor;

    // This isn't quite the right requirement because it imposes
    // convertibility, but it's good enough for a first approximation.
    BOOST_CONCEPT_ASSERT((InteroperableIterator<cursor,end_cursor>));

    ~Sequence()
    {
        elements e = sequence::elements(s);
        cursor c = sequence::begin(s);
        end_cursor e = sequence::end(s);
    }
 private:
    S s;
};

template <class S>
struct LvalueSequence
  : Sequence<S>
{
    BOOST_CONCEPT_ASSERT((LvalueIterator<typename LvalueSequence::cursor>));
};

template <class S>
struct SinglePassSequence
  : Sequence<S>
{
    BOOST_CONCEPT_ASSERT((SinglePassIterator<typename SinglePassSequence::cursor>));
};

template <class S>
struct ForwardSequence
  : SinglePassSequence<S>
{
    BOOST_CONCEPT_ASSERT((ForwardTraversal<typename ForwardSequence::cursor>));
};

template <class S>
struct BidirectionalSequence
  : ForwardSequence<S>
{
    BOOST_CONCEPT_ASSERT((BidirectionalTraversal<typename BidirectionalSequence::cursor>));
};
    
template <class S>
struct RandomAccessSequence
  : BidirectionalSequence<S>
{
    BOOST_CONCEPT_ASSERT((RandomAccessTraversal<typename RandomAccessSequence::cursor>));
};
    
template <class S>
struct MutableSequence
  : Sequence
{
    BOOST_CONCEPT_ASSERT((
        ReadWritePropertyMap<
            typename MutableSequence::elements
          , typename MutableSequence::cursor>));
};


template <class S>
struct Mutable_SinglePassSequence
  : SinglePassSequence<S>
  , MutableSequence<S>
{};

template <class S>
struct Mutable_ForwardSequence
  : MutableSequence<S>
  , ForwardSequence<S>
{};

template <class S>
struct Mutable_BidirectionalSequence
  : MutableSequence<S>
  , BidirectionalSequence<S>
{};

template <class S>
struct Mutable_RandomAccessSequence
  : MutableSequence<S>
  , RandomAccessSequence<S>
{};

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_CONCEPT_DWA200652_HPP
