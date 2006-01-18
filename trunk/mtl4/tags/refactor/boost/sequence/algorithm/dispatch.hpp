// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP

# include <boost/sequence/core/category.hpp>
# include <boost/typeof/typeof.hpp>
# include <boost/mpl/apply_wrap.hpp>
# include <boost/mpl/assert.hpp>
# include <boost/type_traits/is_same.hpp>

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

//
// specialization for two input ranges.
//

# if BOOST_WORKAROUND(BOOST_MSVC, <= 1310)                          \
  || BOOST_WORKAROUND(_MSC_FULL_VER, BOOST_TESTED_AT(140050601))

# pragma warning(push)
# pragma warning(disable: 4675) // warning resolved overload was found by argument-dependent lookup

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

template <class AlgorithmID, class Range1, class Range2>
struct dispatch<AlgorithmID(Range1&,Range2&)>
{
    typedef typename dispatch0<
        AlgorithmID(Range1&,Range2&)
    >::type implementation;

    typedef typename implementation::template apply<Range1,Range2>::type type;
};

# else

template <class AlgorithmID, class Range1, class Range2>
struct dispatch<AlgorithmID(Range1&,Range2&)>
{
    typedef typename category<Range1>::type cat1;
    typedef typename category<Range2>::type cat2;

    // lookup_implementation uses ADL on category tags to look up an
    // implementation class for this algorithm id and sequence
    // categories.  lookup_implementation is the *only* symbol that is
    // subject to ADL in this dispatching scheme.
    
    typedef BOOST_TYPEOF_TPL(
         lookup_implementation(AlgorithmID(), cat1(), cat2())
    ) implementation;

    // GCC3 seems to require the use of apply_wrap; other compilers
    // could access the nested apply template directly.
    typedef typename mpl::apply_wrap2<implementation,Range1,Range2>::type type;
};

# endif

}}} // namespace boost::sequence::algorithm

#endif // BOOST_SEQUENCE_ALGORITHM_DISPATCH_DWA200559_HPP
