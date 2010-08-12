// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/sequence/algorithm/copy.hpp>
#include <boost/test/execution_monitor.hpp>
#include <algorithm>

#include <boost/mpl/assert.hpp>

namespace sequence = boost::sequence;

// See odr.cpp for explanation.
void odr2()
{
    boost::array<char,6> const hello = {{'h','e','l','l','o','\0'}};
    boost::array<char,11> buf = {{'0','1','2','3','4','5','6','7','8','9','\0'}};

    sequence::algorithm::copy(hello, buf);
}

