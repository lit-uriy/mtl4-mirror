// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef IDENTITY_PROPERTY_MAP_DWA200552_HPP
# define IDENTITY_PROPERTY_MAP_DWA200552_HPP

# include <boost/utility/result_of.hpp>

namespace boost { namespace sequence { 

// A property map whose keys and values are the same
struct identity_property_map
{
    // Readability
    template <class K>
    inline
    K& operator()(K& k) const
    {
        return k;
    }

    template <class K>
    inline
    K const& operator()(K const& k) const
    {
        return k;
    }

    // Writability
    template <class K, class V>
    inline
    void operator()(K& k, V const& v) const
    {
        k = v;
    }

    // This one is needed to support proxies
    template <class K, class V>
    inline
    void operator()(K const& k, V const& v) const
    {
        k = v;
    }
};

}} // namespace boost::sequence

namespace boost  {

// lvalues
template <class K>
struct result_of<sequence::identity_property_map(K&)>
{
    typedef K& type;
};

template <class K>
struct result_of<sequence::identity_property_map(K const&)>
{
    typedef K const& type;
};

// rvalues
template <class K>
struct result_of<sequence::identity_property_map(K)>
{
    typedef K const& type;
};

}

#endif // IDENTITY_PROPERTY_MAP_DWA200552_HPP
