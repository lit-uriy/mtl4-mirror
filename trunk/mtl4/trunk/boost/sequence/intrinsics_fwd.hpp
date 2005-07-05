// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP
# define BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP

# include <boost/sequence/tag.hpp>

namespace boost { namespace sequence { 

// This class template can be specialized to provide the
// implementation of intrinsic operations for sequences having a
// particular tag.  Specializations should contain three nested
// classes named begin, end, and elements.  Each one should be a unary
// function object accepting a Sequence parameter and returning the
// result of the correspondingly-named intrinsic function.  Each one
// should also have a nested ::type that indicates the result type of
// its operator().
template <class Sequence, class Tag = typename tag<Sequence>::type>
struct intrinsics;

}} // namespace boost::sequence

#endif // BOOST_SEQUENCE_INTRINSICS_FWD_DWA2005616_HPP
