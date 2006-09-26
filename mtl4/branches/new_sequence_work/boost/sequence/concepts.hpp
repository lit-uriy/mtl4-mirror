// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP
# define BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP

# include <boost/sequence/begin.hpp>
# include <boost/sequence/end.hpp>
# include <boost/sequence/reader.hpp>
# include <boost/sequence/writer.hpp>
# include <boost/sequence/insert.hpp>
# include <boost/type_traits/add_reference.hpp>
# include <boost/sequence/traits/size_type.hpp>
# include <boost/sequence/traits/index_type.hpp>

namespace boost { namespace sequence { namespace concepts { 

using namespace boost::cursor::concepts;

template <class S>
struct Sequence
{
    // The end cursor doesn't have to have the same type as the begin
    // cursor, just as long as you can compare them.
    typedef typename result_of<
        op::begin(typename add_reference<S>::type)
    >::type begin_cursor;
    
    typedef typename result_of<
        op::end(typename add_reference<S>::type)
    >::type end_cursor;

    typedef mpl::not_< is_incrementable<begin_cursor> > has_static_size;

    BOOST_CONCEPT_ASSERT((SinglePassCursor<begin_cursor>));
    BOOST_CONCEPT_ASSERT((Cursor<end_cursor>));
    BOOST_CONCEPT_ASSERT((EqualityComparable<begin_cursor,end_cursor>));
    
    BOOST_CONCEPT_USAGE(Sequence)
    {
        begin_cursor b = sequence::begin(s);
        end_cursor e = sequence::end(s);
        
        ignore_unused_variable_warning(b);
        ignore_unused_variable_warning(e);
    }
    
 private:
    S s;
};

template <class S>
struct ReadableSequence
  : Sequence<S>
{
    typedef typename result_of<
        op::reader(typename add_reference<S>::type)
    >::type reader;

    typedef typename result_of<
        reader(typename op
               
    BOOST_CONCEPT_ASSERT((CopyConstructible<reader>));
    BOOST_CONCEPT_ASSERT((Assignable<reader>));
    
    BOOST_CONCEPT_USAGE(ReadableSequence)
    {
        typedef typename ReadableSequence::has_static_size is_static;
        test_reader(is_static());
    }
    
 private:
    void test_reader(mpl::false_ is_static)
    {
        reader r = sequence::reader(s);
        r(deref(begin(s)));
        BOOST_CONCEPT_ASSERT((UnaryFunction<reader,
    }
        
    S s;
};


template <class S>
struct WritableSequence
  : Sequence<S>
{
    typedef typename result_of<
        op::writer(typename add_reference<S>::type)
    >::type writer;

    BOOST_CONCEPT_ASSERT((CopyConstructible<writer>));
    BOOST_CONCEPT_ASSERT((Assignable<writer>));
    
    BOOST_CONCEPT_USAGE(WritableSequence)
    {
        writer w = sequence::writer(s);
        ignore_unused_variable_warning(w);
    }
    
 private:
    S s;
};

template <class S>
struct O1SizeSequence
  : Sequence<S>
{
    // A type representing the size of the sequence.
    typedef typename sequence::traits::size_type<S>::type size_type;

    // A type that can act as an index into the sequence.  Because
    // size_type may be a wrapper for a compile-time constant,
    // e.g. mpl::size_t<N>, we need a distinct type in order to be
    // able to count at runtime.
    typedef typename sequence::traits::index_type<S>::type index_type;
    
    ~O1SizeSequence()
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
    S s;
};

template <class S>
struct InsertableSequence
  : Sequence<S>
{
    ~InsertableSequence()
    {
        typename InsertableSequence::cursor pos;
        pos = sequence::insert(s, pos, v);
    }
    
 private:
    S s;
    typename InsertableSequence::value_type const v;
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
struct Mutable_Sequence
  : Sequence<S>
{
    BOOST_CONCEPT_ASSERT((
        ReadWritePropertyMap<
            typename Mutable_Sequence::elements
          , typename Mutable_Sequence::cursor>));
};


template <class S>
struct Mutable_SinglePassSequence
  : SinglePassSequence<S>
  , Mutable_Sequence<S>
{};

template <class S>
struct Mutable_ForwardSequence
  : Mutable_Sequence<S>
  , ForwardSequence<S>
{};

template <class S>
struct Mutable_BidirectionalSequence
  : Mutable_Sequence<S>
  , BidirectionalSequence<S>
{};

template <class S>
struct Mutable_RandomAccessSequence
  : Mutable_Sequence<S>
  , RandomAccessSequence<S>
{};

}}} // namespace boost::sequence::concepts

#endif // BOOST_SEQUENCE_CONCEPTS_DWA200652_HPP
