// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/sequence/algorithm/copy.hpp>
#include <boost/test/minimal.hpp>
//#include <boost/array.hpp>
#include <algorithm>

namespace sequence = boost::sequence;

template <class X> struct unknown;

int test_main( int argc, char* argv[] )
{
    char const hello[] = "hello";
    char buf[] = "0123456789";
    char buf2[10];

//    boost::array<char,6> const hello = {{'h', 'e', 'l', 'l', 'o'}};
//    boost::array<char,11> buf = {{'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 0 }};
//    boost::array<char, 10> buf2;
    
    sequence::algorithm::copy(
        sequence::algorithm::copy(hello, buf)
      , buf2
    );

    BOOST_REQUIRE(
        std::equal(buf,buf+sizeof(buf),"hello\06789\0")
    );
    BOOST_REQUIRE(
        std::equal(buf2,&buf2[0]+5,"6789")
    );
}
