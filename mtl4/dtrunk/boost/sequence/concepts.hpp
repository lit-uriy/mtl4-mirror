// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP
# define BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP

# include <boost/property_map/concepts.hpp>
# include <boost/sequence/begin.hpp>
# include <boost/sequence/end.hpp>
# include <boost/sequence/elements.hpp>
# include <boost/type_traits/add_reference.hpp>
# include <boost/sequence/traits/size_type.hpp>
# include <boost/sequence/traits/index_type.hpp>

namespace boost { namespace sequence { namespace concepts { 

using namespace boost::property_map::concepts;

template <class S>
struct Sequence
  : ReadablePropertyMap<   
        typename result_of<
            // Note that we *must* add_reference to S because it might
            // be an array type that would otherwise decay into a
            // pointer.
            op::elements(typename add_reference<S>::type)
        >::type
      , typename result_of<
            op::begin(typename add_reference<S>::type)
        >::type
    >
{
    // Associated types cursor, elements, key_type, value_type,
    // and reference, all come from ReadablePropertyMap

    // The end cursor doesn't have to have the same type as the begin
    // cursor, just as long as you can compare them.
    typedef typename result_of<
        op::end(typename add_reference<S>::type)
    >::type  end_cursor;

    // This isn't quite the right requirement because it imposes
    // convertibility, but it's good enough for a first approximation.
    BOOST_CONCEPT_ASSERT((
        InteroperableIterator<typename Sequence::cursor,end_cursor>));

    ~Sequence()
    {
        typename Sequence::elements elts = sequence::elements(s);
        typename Sequence::cursor c = sequence::begin(s);
        end_cursor end = sequence::end(s);
    }
 private:
    S s;
};

template <class S>
struct O1SizeSequence
  : Sequence<S>
{
    typedef typename sequence::size_type<S>::type size_type;
    typedef typename sequence::index_type<S>::type index_type;
    
    ~Sequence()
    {
        size_type size = sequence::size(s);

        BOOST_CONCEPT_ASSERT((Convertible<size_type,index_type>));

        index_type i = sequence::size(s);
        
        // Need EqualityComparable2
        i == size;
        i != size;
        size == i;
        size != i;
    }
 private:
    Sequence s;
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
  : Sequence<S>
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

}}} // namespace boost::sequence::concepts

#endif // BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP
