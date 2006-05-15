// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_PROPERTY_MAP_IDENTITY_DWA200657_HPP
# define BOOST_PROPERTY_MAP_IDENTITY_DWA200657_HPP

# include <boost/type_traits/add_const.hpp>
# include <boost/type_traits/add_reference.hpp>

namespace boost { namespace property_map { 

struct identity
{
    // For result_of support
    template<class K>
    struct result;

    // Return references unchanged.  
    // Add const& to all non-references.
    template<class This, class K>
    struct result<This(K)>
      : add_reference< typename add_const<K>::type >
    {};

    template<class This, class K, class V>
    struct result<This(K,V)>
    {
        typedef void type;
    };

    template <class K>
    K const& operator()(K const& k) const
    {
        return k;
    }

    template <class K>
    K& operator()(K& k) const
    {
        return k;
    }

    template <class K, class V>
    void operator()(K& k, V& v) const
    {
        k = v;
    }

    template <class K, class V>
    void operator()(K& k, V const& v) const
    {
        k = v;
    }

    template <class K, class V>
    void operator()(K const& k, V& v) const
    {
        k = v;
    }

    template <class K, class V>
    void operator()(K const& k, V const& v) const
    {
        k = v;
    }
};

}} // namespace boost::property_map

#endif // BOOST_PROPERTY_MAP_IDENTITY_DWA200657_HPP
