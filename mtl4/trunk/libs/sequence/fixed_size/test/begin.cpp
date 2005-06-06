// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/assert.hpp>
#include <boost/sequence/begin_cursor.hpp>
#include <boost/sequence/detail/is_specialized.hpp>
#include <boost/sequence/begin.hpp>

namespace sequence = boost::sequence;

BOOST_MPL_ASSERT(
    (sequence::detail::is_specialized<
         sequence::begin_cursor<
             char const[6]
         >
     >));

BOOST_MPL_ASSERT(
    (sequence::detail::is_specialized<
         sequence::begin_cursor<
             char[6]
         >
     >));

int main()
{
    sequence::begin("hello");
}
