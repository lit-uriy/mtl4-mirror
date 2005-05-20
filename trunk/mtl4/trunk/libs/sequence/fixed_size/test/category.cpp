// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#include <boost/mpl/assert.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/sequence/category.hpp>

BOOST_MPL_ASSERT(
    (boost::is_same<
         typename boost::sequence::category<char const[6]>::type
       , boost::sequence::algorithm::fixed_size::category
     >));

