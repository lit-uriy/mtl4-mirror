// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_UNROLLED_DISPATCH_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_UNROLLED_DISPATCH_DWA200559_HPP

# include <boost/typeof/typeof.hpp>
# include <boost/sequence/core/category_fwd.hpp>
# include <boost/sequence/fixed_size/category.hpp>

#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace boost { namespace sequence {

namespace algorithm
{
  // A strategy tag that can be specialized to provide implementations
  // for any given AlgorithmID using that strategy.
  template <class AlgorithmID>
  struct unrolled {};
}

namespace fixed_size
{
  // Rule: any algorithm whose source sequence is fixed size will be
  // unrolled (unless there is a more specific rule for that case).
  template <class Algorithm, class TargetCategory>
  algorithm::unrolled<Algorithm>
  lookup_implementation(Algorithm,sequence::fixed_size::category,TargetCategory)
  {
      return algorithm::unrolled<Algorithm>();
  }
}

}} // namespace boost::sequence::algorithm::unrolled

// Strategies need to be available to typeof, so we register the
// unrolled template.
BOOST_TYPEOF_REGISTER_TEMPLATE(boost::sequence::algorithm::unrolled,(class))

#endif // BOOST_SEQUENCE_ALGORITHM_UNROLLED_DISPATCH_DWA200559_HPP
