// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/sequence/concepts.hpp>
#include <boost/sequence/composed.hpp>
#include <list>
#include <vector>

int main()
{
    using namespace boost::sequence::concepts;
    using boost::sequence::composed;
    
    BOOST_CONCEPT_ASSERT((LvalueSequence<int const[4]>));
    BOOST_CONCEPT_ASSERT((LvalueSequence<int[4]>));
    
    BOOST_CONCEPT_ASSERT((RandomAccessSequence<int const[4]>));
    BOOST_CONCEPT_ASSERT((Mutable_RandomAccessSequence<int[4]>));
    
    BOOST_CONCEPT_ASSERT((BidirectionalSequence<std::list<int> const>));
    BOOST_CONCEPT_ASSERT((Mutable_BidirectionalSequence<std::list<int> >));

    BOOST_CONCEPT_ASSERT((BidirectionalSequence<std::vector<int> const>));
    BOOST_CONCEPT_ASSERT((RandomAccessSequence<std::vector<int> const>));
    BOOST_CONCEPT_ASSERT((Mutable_RandomAccessSequence<std::vector<int> >));

    // These tests verify that proxy references work
    BOOST_CONCEPT_ASSERT((RandomAccessSequence<std::vector<bool> const>));
    BOOST_CONCEPT_ASSERT((Mutable_RandomAccessSequence<std::vector<bool> >));

    BOOST_CONCEPT_ASSERT((RandomAccessSequence<composed<char const*> >));
    BOOST_CONCEPT_ASSERT((Mutable_RandomAccessSequence<composed<char *> >));
}
