// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_UNROLLED_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_UNROLLED_DWA200559_HPP

# include <boost/typeof/typeof.hpp>
# include <boost/sequence/algorithm/fixed_size/category.hpp>

#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

template <class AlgorithmID>
struct unrolled {};

BOOST_TYPEOF_REGISTER_TEMPLATE(unrolled,1)

template <class Algorithm>
unrolled<Algorithm>
lookup_implementation(Algorithm,fixed_size::category,fixed_size::category)
{
    return unrolled<Algorithm>();
}

}}}} // namespace boost::sequence::algorithm::fixed_size

#endif // BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_UNROLLED_DWA200559_HPP
