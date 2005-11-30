// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/sequence/minus.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost::sequence;
namespace mpl = boost::mpl;

int test_main(int, char*[] )
{
    BOOST_REQUIRE(
        minus(3,5) == (3-5)
    );

    BOOST_REQUIRE(
        minus(mpl::int_<3>(), mpl::int_<5>()) == mpl::int_<(3-5)>()
    );

    BOOST_REQUIRE(
        minus(mpl::int_<3>(), 5) == 3-5
    );

    BOOST_REQUIRE(
        minus(3, mpl::int_<5>()) == 3-5
    );
    
    return 0;
}
