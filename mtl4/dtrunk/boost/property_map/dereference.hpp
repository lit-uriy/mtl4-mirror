// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_PROPERTY_MAP_DEREFERENCE_DWA200655_HPP
# define BOOST_PROPERTY_MAP_DEREFERENCE_DWA200655_HPP

# include <boost/iterator/iterator_concepts.hpp>
# include <boost/concept/where.hpp>
# include <boost/type_traits/remove_reference.hpp>
# include <boost/type_traits/remove_cv.hpp>

namespace boost { namespace property_map { 

using namespace boost_concepts;

struct dereference
{
    // For result_of support
    template<class K>
    struct result;

    template<class This, class I>
    struct result<This(I)>
    {
        typedef typename remove_cv<
           typename remove_reference<I>::type
        >::type iter;
        
        typedef typename ReadableIterator<iter>::reference type;
    };

    template<class This, class K, class V>
    struct result<This(K,V)>
    {
        typedef void type;
    };
    
    template <class I>
    BOOST_CONCEPT_WHERE(
        ((ReadableIterator<I>)),
        
    (typename ReadableIterator<I>::reference))
    operator()(I const& i) const
    {
        return *i;
    }

    template <class I, class V>
    BOOST_CONCEPT_WHERE(
        ((WritableIterator<I>)),
        
    (void))
    operator()(I const& i, V const& v) const
    {
        *i = v;
    }


    template <class I, class V>
    BOOST_CONCEPT_WHERE(
        ((WritableIterator<I>)),
        
    (void))
    operator()(I const& i, V& v) const
    {
        *i = v;
    }
};

}} // namespace boost::property_map

#endif // BOOST_PROPERTY_MAP_DEREFERENCE_DWA200655_HPP
