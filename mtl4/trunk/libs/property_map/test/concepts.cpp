// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <list>
#include <boost/property_map/concept.hpp>
#include <boost/property_map/identity.hpp>
#include <boost/property_map/dereference.hpp>
#include <boost/iterator/counting_iterator.hpp>

int main()
{
    namespace pm = boost::property_map;
    
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::identity,int*>));
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::identity,int const*>));
    BOOST_CONCEPT_ASSERT((pm::WritablePropertyMap<pm::identity,int*>));
    BOOST_CONCEPT_ASSERT((pm::ReadWritePropertyMap<pm::identity,int*>));
#if CONCEPT_FAIL == 1
    BOOST_CONCEPT_ASSERT((pm::WritablePropertyMap<pm::identity,int const*>));
#elif CONCEPT_FAIL == 2 
    BOOST_CONCEPT_ASSERT((pm::ReadWritePropertyMap<pm::identity,int const*>));
#endif 
    
    typedef std::list<int>::iterator li;
    typedef std::list<int>::const_iterator cli;
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::identity, li>));
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::identity, cli>));
    BOOST_CONCEPT_ASSERT((pm::WritablePropertyMap<pm::identity, li>));
    BOOST_CONCEPT_ASSERT((pm::ReadWritePropertyMap<pm::identity, li>));
#if CONCEPT_FAIL == 3
    BOOST_CONCEPT_ASSERT((pm::WritablePropertyMap<pm::identity, cli>));
#elif CONCEPT_FAIL == 4 
    BOOST_CONCEPT_ASSERT((pm::ReadWritePropertyMap<pm::identity, cli>));
#endif 

    typedef boost::counting_iterator<li> li2;
    typedef boost::counting_iterator<cli> cli2;
    
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::dereference, li2>));
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::dereference, cli2>));
    BOOST_CONCEPT_ASSERT((pm::WritablePropertyMap<pm::dereference, li2, int>));
    BOOST_CONCEPT_ASSERT((pm::ReadWritePropertyMap<pm::dereference, li2>));
    
#if CONCEPT_FAIL == 5
    BOOST_CONCEPT_ASSERT((pm::ReadablePropertyMap<pm::dereference,int*>));
#endif 

}
