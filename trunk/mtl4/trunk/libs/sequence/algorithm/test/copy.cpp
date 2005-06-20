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
           sequence::fixed_size_indexable_tag<11>
         ,sequence::tag<boost::array<int,11> >::type
         >
        ));

#if 1  // help in tracking down some bugs
    boost::array<int,3> x;
    sequence::begin(x);
    
    sequence::begin(hello);
    sequence::begin(buf);
    char const (&hello_)[6] = hello;
    char (&buf_)[11] = buf;
    sequence::begin(hello_);
    sequence::begin(buf_);
#endif 

    BOOST_MPL_ASSERT(
        (boost::is_same<
           sequence::fixed_size_indexable_tag<11>
          ,sequence::tag<char[11]>::type
         >
        ));

    BOOST_MPL_ASSERT(
        (boost::is_same<
             sequence::fixed_size::cursor<3>
           , sequence::successor<
                 sequence::fixed_size::cursor<2>
             >::type
          >
        ));


    BOOST_MPL_ASSERT(
        (boost::is_same<
             sequence::fixed_size::cursor<3>
           , sequence::advanced<
                 sequence::fixed_size::cursor<0>
               , boost::mpl::size_t<3>
             >::type
          >
        ));
#if 1
    {
        boost::array<char,6> const hello = {{'h','e','l','l','o','\0'}};
        boost::array<char,11> buf = {{'0','1','2','3','4','5','6','7','8','9','\0'}};
        boost::array<char,11> buf2;
        
        sequence::algorithm::copy(hello, buf);
    
        sequence::algorithm::copy(
            sequence::algorithm::copy(hello, buf)
          , buf2
        );

        boost::array<char,11> const result = {{'h','e','l','l','o','\0','6','7','8','9','\0'}};
        BOOST_REQUIRE(
            std::equal(&buf[0],&buf[0]+buf.size(), &result[0] )
        );
        BOOST_REQUIRE(
            std::equal(&buf2[0],&buf2[0]+5,"6789")
        );
    }
#endif
    
    return 0;
}
