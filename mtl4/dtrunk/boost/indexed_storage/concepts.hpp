// Copyright David Abrahams 2006. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP
# define BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP

# include <boost/sequence/concepts.hpp>

namespace boost { namespace indexed_storage { namespace concepts {

using namespace boost::sequence::concepts;

template <class S>
struct IndexedStorage
  : Sequence<S>
{
    typedef typename result_of<
        op::indices(typename add_reference<S>::type)
    >::type indices;

    BOOST_CONCEPT_ASSERT((ReadablePropertyMap<indices,IndexedStorage::cursor>));

    typedef typename
        ReadablePropertyMap<indices,IndexedStorage::cursor>::value_type
    index_type;

    BOOST_CONCEPT_ASSERT((UnsignedInteger<index_type>));
};

}}} // namespace boost::indexed_storage::concepts

#endif // BOOST_INDEXED_STORAGE_CONCEPTS_DWA200658_HPP
