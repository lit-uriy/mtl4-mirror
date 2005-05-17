// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef INDEX_PROPERTY_MAP_DWA200552_HPP
# define INDEX_PROPERTY_MAP_DWA200552_HPP

# include <boost/utility/result_of.hpp>

namespace boost { namespace sequence { 

// A property map for accessing indexable objects
template <class Indexable>
struct index_property_map
{
    index_property_map(Indexable index)
      : index(index)
    {}
    
    // Readability
    template <class K>
    K&
    inline operator()(K& k) const
    {
        return index[k];
    }

    template <class K>
    K const&
    inline operator()(K const& k) const
    {
        return k;
    }

    // Writability
    template <class K, class V>
    void
    inline operator()(K& k, V const& v) const
    {
        return k = v;
    }

    // This one is needed to support proxies
    template <class K, class V>
    void
    inline operator()(K const& k, V const& v) const
    {
        return k = v;
    }

    Indexable index;
};

}} // namespace boost::sequence

namespace boost  {

// lvalues
template <class K>
struct result_of<sequence::index_property_map(K&)>
{
    typedef K& type;
};

template <class K>
struct result_of<sequence::index_property_map(K const&)>
{
    typedef K const& type;
};

// rvalues
template <class K>
struct result_of<sequence::index_property_map(K)>
{
    typedef K const& type;
};

}

#endif // INDEX_PROPERTY_MAP_DWA200552_HPP
