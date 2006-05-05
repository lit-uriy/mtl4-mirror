// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_OPERATIONS_ELEMENTS_JDG20060207_HPP
# define BOOST_SEQUENCE_OPERATIONS_ELEMENTS_JDG20060207_HPP

# include <boost/sequence/operations/iterator_range_operations.hpp>

namespace boost { namespace sequence { namespace intrinsic {

// The default implementation of each intrinsic function object type
// is inherited from the corresponding member of
// operations<Sequence>.  You can of course specialize begin<S>,
// end<S>, and elements<S>, individually, but specializing
// operations<> usually more convenient.

template <class Sequence>
struct elements : operations<Sequence>::elements {};

}}} // namespace boost::sequence::intrinsic

#endif // BOOST_SEQUENCE_OPERATIONS_ELEMENTS_JDG20060207_HPP
