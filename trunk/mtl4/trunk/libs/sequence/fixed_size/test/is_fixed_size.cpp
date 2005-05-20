// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/mpl/assert.hpp>
#include <boost/sequence/fixed_size/is_fixed_size.hpp>

BOOST_MPL_ASSERT((boost::sequence::fixed_size::is_fixed_size<char const[6]>));
