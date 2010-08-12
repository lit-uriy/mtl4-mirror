// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/sequence/algorithm/copy.hpp>
#include <boost/test/minimal.hpp>
#include <algorithm>

#include <boost/mpl/assert.hpp>

namespace sequence = boost::sequence;

int test_main( int argc, char* argv[] )
{
    char const hello[] = "hello";
    char buf[] = "0123456789";

    BOOST_MPL_ASSERT(
        (boost::is_same<
         sequence::fixed_size::tag<11>
         ,sequence::intrinsic::tag<boost::array<int,11> >::type
         >
        ));

    BOOST_MPL_ASSERT(
        (boost::is_same<
         sequence::fixed_size::tag<11>
         ,sequence::intrinsic::tag<char[11]>::type
         >
        ));

    BOOST_MPL_ASSERT(
        (boost::is_same<
         sequence::fixed_size::cursor<3>
         , sequence::intrinsic::next<
         sequence::fixed_size::cursor<2>
         >::type
         >
        ));


    BOOST_MPL_ASSERT(
        (boost::is_same<
         sequence::fixed_size::cursor<3>
         , sequence::intrinsic::advance<
         sequence::fixed_size::cursor<0>
         , boost::mpl::size_t<3>
         >::type
         >
        ));
    
    char buf2[11];

    sequence::algorithm::copy(hello, buf);
    
    sequence::algorithm::copy(
        sequence::algorithm::copy(hello, buf)
      , buf2
    );

    boost::array<char,11> const result = {{'h','e','l','l','o','\0','6','7','8','9','\0'}};
        
    BOOST_REQUIRE(
        std::equal(&buf[0],&buf[0]+sizeof(buf), &result[0] )
    );
    BOOST_REQUIRE(
        std::equal(&buf2[0],&buf2[0]+5,"6789")
    );
    
    return 0;
}

