// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP
# define BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP

# include <boost/sequence/concepts.hpp>
# include <boost/type_traits/add_reference.hpp>
# include <boost/indexed_storage/indices.hpp>
# include <boost/indexed_storage/get_at.hpp>
# include <boost/indexed_storage/set_at.hpp>

namespace boost { namespace indexed_storage { namespace concepts {

using namespace boost::sequence::concepts;
using boost::sequence::concepts::Sequence;

template <class S>
struct IndexedStorage
  : Sequence<S>
{
    typedef typename result_of<
        op::indices(typename add_reference<S>::type)
    >::type indices;

    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<indices,typename IndexedStorage::cursor>));

    typedef typename
        ReadablePropertyMap<indices,typename IndexedStorage::cursor>::value_type
    index_type;

    BOOST_CONCEPT_ASSERT((Integer<index_type>));

    ~IndexedStorage()
    {
        typename IndexedStorage::value_type v = get_at(s, i);
        ignore_unused_variable_warning(v);
    }
 private:
    S s;
    index_type i;
};

typedef ::boost::indexed_storage::concepts::IndexedStorage<int> foo;

template <class S>
struct Mutable_IndexedStorage
  : IndexedStorage<S>
  , Mutable_Sequence<S>
{
    ~Mutable_IndexedStorage()
    {
        set_at(s, i, v);
    }
 private:
    S s;
    typename Mutable_IndexedStorage::index_type i;
    typename Mutable_IndexedStorage::value_type v;
};

}}} // namespace boost::indexed_storage::concepts

#endif // BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP
