// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/indexed_storage/concepts.hpp>
#include <boost/sequence/composed.hpp>
#include <list>
#include <vector>

int main()
{
    using namespace boost::indexed_storage::concepts;
    using boost::sequence::composed;
    
    BOOST_CONCEPT_ASSERT((IndexedStorage<int const[4]>));
    BOOST_CONCEPT_ASSERT((Mutable_IndexedStorage<int[4]>));
    
    BOOST_CONCEPT_ASSERT((IndexedStorage<std::vector<int> const>));
    BOOST_CONCEPT_ASSERT((Mutable_IndexedStorage<std::vector<int> >));

    BOOST_CONCEPT_ASSERT((IndexedStorage<composed<char const*> >));
    BOOST_CONCEPT_ASSERT((Mutable_IndexedStorage<composed<char *> >));
}
