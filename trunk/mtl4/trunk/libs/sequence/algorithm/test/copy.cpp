// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if 0

# include <boost/sequence/begin_cursor.hpp>

boost::sequence::begin_cursor<char[11]> x;

#else

#include <boost/sequence/algorithm/copy.hpp>
#include <boost/test/minimal.hpp>
#include <algorithm>

namespace sequence = boost::sequence;

int test_main( int argc, char* argv[] )
{
    char const hello[] = "hello";
    char buf[] = "0123456789";
    char buf2[11];

#if 0  // help in tracking down some bugs
    sequence::begin(hello);
    sequence::begin(buf);
    char const (&hello_)[6] = hello;
    char (&buf_)[11] = buf;
    sequence::begin(hello_);
    sequence::begin(buf_);
#endif 
    
    sequence::algorithm::copy(hello, buf);
    
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

    return 0;
}

#endif
