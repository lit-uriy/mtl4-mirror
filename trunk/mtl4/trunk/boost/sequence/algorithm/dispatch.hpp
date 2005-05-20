// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP

# include <boost/sequence/category.hpp>
# include <boost/sequence/algorithm/fixed_size/category.hpp>
# include <boost/typeof/typeof.hpp>

#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()

namespace boost { namespace sequence { namespace algorithm { 

// When instantiated as
//
//   dispatch< AlgorithmID(Range1, Range2, ... RangeN) >
//
// will contain two type members:
//
//   implementation - a class containing a N-ary ::execute() static
//     member function [template] that can be invoked on arguments of
//     type Range1, ... RangeN to perform the requested algorithm.
//
//   type - the result type of that ::execute member
//
template <class Signature> struct dispatch;

# if !(BOOST_MSVC <= 1301) && BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050215))
namespace aux_
{
  struct dummy { template <class T1, class T2> struct apply {}; };
}
char lookup_implementation(...);

template <class Signature> struct dispatch0;
template <class AlgorithmID, class Range1, class Range2>
struct dispatch0<AlgorithmID(Range1&,Range2&)>
{
    typedef typename sequence::category<Range1>::type cat1;
    typedef typename sequence::category<Range2>::type cat2;

    // lookup_implementation uses ADL on category tags to look up an
    // implementation class for this algorithm id and sequence
    // categories.  lookup_implementation is the *only* symbol that is
    // subject to ADL in this dispatching scheme.
    typedef BOOST_TYPEOF_TPL(
         lookup_implementation(AlgorithmID(), cat1(), cat2())
    ) type;
};
# endif

// specialization for two input ranges.
template <class AlgorithmID, class Range1, class Range2>
struct dispatch<AlgorithmID(Range1&,Range2&)>
{
# if  BOOST_MSVC <= 1301 || !BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050215))
    typedef typename category<Range1>::type cat1;
    typedef typename category<Range2>::type cat2;

    // lookup_implementation uses ADL on category tags to look up an
    // implementation class for this algorithm id and sequence
    // categories.  lookup_implementation is the *only* symbol that is
    // subject to ADL in this dispatching scheme.
    typedef BOOST_TYPEOF_TPL(
         lookup_implementation(AlgorithmID(), cat1(), cat2())
    ) implementation_;
# else
    typedef typename dispatch0<
        AlgorithmID(Range1&,Range2&)
    >::type implementation_;
# endif 

    typedef typename
      implementation_::template apply<Range1,Range2>::type
    type;
};

}}} // namespace boost::sequence::algorithm

#endif // BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP
