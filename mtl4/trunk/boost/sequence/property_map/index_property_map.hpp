// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef INDEX_PROPERTY_MAP_DWA200552_HPP
# define INDEX_PROPERTY_MAP_DWA200552_HPP

# include <boost/sequence/core/detail/o1_size_cursors.hpp>
# include <boost/utility/result_of.hpp>

namespace boost { namespace sequence { 

# ifdef BOOST_MSVC
#  pragma warning(push)
#  pragma warning(disable:4512) // assignment operator could not be generated
# endif

// A property map for accessing indexable objects
template <class Indexable>
struct index_property_map
{
    index_property_map(Indexable index)
      : index(index)
    {}
    
    // Readability
    template <class K>
    inline
    typename detail::index_result<Indexable,K&>::type
    operator()(K& k) const
    {
        return index[k];
    }

    template <class K>
    inline
    typename detail::index_result<Indexable,K const&>::type
    operator()(K const& k) const
    {
        return index[k];
    }

    // Writability
    template <class K, class V>
    inline
    void operator()(K& k, V const& v) const
    {
        index[k] = v;
    }

    // This one is needed to support proxies
    template <class K, class V>
    inline
    void operator()(K const& k, V const& v) const
    {
        index[k] = v;
    }

    Indexable index;
};

# ifdef BOOST_MSVC
#  pragma warning(pop)
# endif


}} // namespace boost::sequence

namespace boost  {

// lvalues
template <class Indexable, class K>
struct result_of<sequence::index_property_map<Indexable>(K&)>
  : sequence::detail::index_result<Indexable,K&>
{};

template <class Indexable, class K>
struct result_of<sequence::index_property_map<Indexable>(K const&)>
  : sequence::detail::index_result<Indexable,K const&>
{};

// rvalues
template <class Indexable, class K>
struct result_of<sequence::index_property_map<Indexable>(K)>
  : sequence::detail::index_result<Indexable,K const&>
{};

}

#endif // INDEX_PROPERTY_MAP_DWA200552_HPP
