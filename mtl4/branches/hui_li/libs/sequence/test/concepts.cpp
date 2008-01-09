// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/sequence/concept.hpp>
#include <list>
#include <vector>

int main()
{
    namespace seq = boost::sequence;

    BOOST_CONCEPT_ASSERT((seq::LvalueSequence<int const[4]>));
    BOOST_CONCEPT_ASSERT((seq::LvalueSequence<int[4]>));
    
    BOOST_CONCEPT_ASSERT((seq::RandomAccessSequence<int const[4]>));
    BOOST_CONCEPT_ASSERT((seq::Mutable_RandomAccessSequence<int[4]>));
    
    BOOST_CONCEPT_ASSERT((seq::BidirectionalSequence<std::list<int> const>));
    BOOST_CONCEPT_ASSERT((seq::Mutable_BidirectionalSequence<std::list<int> >));

    BOOST_CONCEPT_ASSERT((seq::BidirectionalSequence<std::vector<int> const>));
    BOOST_CONCEPT_ASSERT((seq::RandomAccessSequence<std::vector<int> const>));
    BOOST_CONCEPT_ASSERT((seq::Mutable_RandomAccessSequence<std::vector<int> >));

    // These tests verify that proxy references work
    BOOST_CONCEPT_ASSERT((seq::RandomAccessSequence<std::vector<bool> const>));
    BOOST_CONCEPT_ASSERT((seq::Mutable_RandomAccessSequence<std::vector<bool> >));
}
