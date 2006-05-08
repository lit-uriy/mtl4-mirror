// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <list>
#include <vector>
#include <boost/property_map/concepts.hpp>
#include <boost/property_map/identity.hpp>
#include <boost/property_map/dereference.hpp>
#include <boost/iterator/counting_iterator.hpp>

int main()
{
    using namespace boost::property_map::concepts;
    using boost::property_map::identity;
    using boost::property_map::dereference;
    
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<identity,int*>));
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<identity,int const*>));
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<identity,int*>));
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<identity,int*>));
#if CONCEPT_FAIL == 1
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<identity,int const*>));
#elif CONCEPT_FAIL == 2 
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<identity,int const*>));
#endif 
    
    typedef std::list<int>::iterator li;
    typedef std::list<int>::const_iterator cli;
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<identity, li>));
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<identity, cli>));
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<identity, li>));
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<identity, li>));
#if CONCEPT_FAIL == 3
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<identity, cli>));
#elif CONCEPT_FAIL == 4 
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<identity, cli>));
#endif 

    typedef boost::counting_iterator<li> li2;
    typedef boost::counting_iterator<cli> cli2;
    
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<dereference, li2>));
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<dereference, cli2>));
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<dereference, li2, int>));
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<dereference, li2>));
    
#if CONCEPT_FAIL == 5
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<dereference,int*>));
#endif 
    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<identity,std::vector<bool>::const_iterator>));
    BOOST_CONCEPT_ASSERT((ReadWritePropertyMap<identity,std::vector<bool>::iterator>));
    BOOST_CONCEPT_ASSERT((WritablePropertyMap<identity,std::vector<bool>::iterator>));
}
