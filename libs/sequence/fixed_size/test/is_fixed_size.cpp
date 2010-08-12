// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/assert.hpp>
#include <boost/sequence/operations/fixed_size/is_fixed_size.hpp>
#include <boost/sequence/operations/fixed_size/cursor.hpp>
#include <boost/sequence/core/property_map/index_utility/property_map.hpp>
#include <boost/sequence/class/range/range.hpp>

namespace sequence = boost::sequence;

BOOST_MPL_ASSERT((sequence::fixed_size::is_fixed_size<char const[6]>));

BOOST_MPL_ASSERT(
    (
        sequence::fixed_size::is_fixed_size<
            sequence::range<
                sequence::index_property_map< char const[6] >
              , sequence::fixed_size::cursor<0>
              , sequence::fixed_size::cursor<1>
            >
        >
    )
    );
