// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/indexed_storage/concepts.hpp>
#include <boost/indexed_storage/sparse.hpp>
#include <vector>
#include <list>
#include <utility>

int main()
{
    using namespace boost::indexed_storage;
    using namespace concepts;

    BOOST_CONCEPT_ASSERT(
        (boost::sequence::concepts::Sequence<
             sparse<std::vector<std::pair<unsigned,float> > > >
        )
    );
                             
    BOOST_CONCEPT_ASSERT((
         Mutable_IndexedStorage<
             sparse<std::vector<std::pair<unsigned,float> > >
         >));

    BOOST_CONCEPT_ASSERT((
         IndexedStorage<
             sparse<std::vector<std::pair<unsigned,float> > > const
         >));

#if 0 // not ready to support this yet.
    BOOST_CONCEPT_ASSERT((
         IndexedStorage<
             sparse<std::vector<std::pair<unsigned,float> > const&>
         >));
#endif 
                             
}
