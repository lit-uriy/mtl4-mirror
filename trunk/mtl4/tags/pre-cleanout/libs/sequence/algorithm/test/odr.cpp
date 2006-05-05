// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/sequence/algorithm/copy.hpp>
#include <boost/test/execution_monitor.hpp>
#include <algorithm>

#include <boost/mpl/assert.hpp>

namespace sequence = boost::sequence;

extern void odr2();

// This test verifies that the techniques we are using to make
// function objects available in all translation units do not cause
// ODR problems in practice.  Technically there are no ODR violations
// in the code, other than in workarounds for broken compilers.
int main( int argc, char* argv[] )
{
    odr2();
    boost::array<char,6> const hello = {{'h','e','l','l','o','\0'}};
    boost::array<char,11> buf = {{'0','1','2','3','4','5','6','7','8','9','\0'}};

    sequence::algorithm::copy(hello, buf);
    return 0;
}
