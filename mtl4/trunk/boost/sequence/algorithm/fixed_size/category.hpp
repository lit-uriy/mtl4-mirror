// Copyright David Abrahams 2005. Distributed under the Boost
// Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
#ifndef BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_CATEGORY_DWA200559_HPP
# define BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_CATEGORY_DWA200559_HPP

namespace boost { namespace sequence { namespace algorithm { namespace fixed_size { 

// Any fixed-size sequence
struct category
{
    typedef category type; // self-returning, for convenience
};

// A fixed-size sequence that can be accessed by a single cursor type.
// For example, even though the normal cursor type for an array varies
// as we traverse the array (so that its position can be represented
// at compile-time), we can also use a pointer as a cursor.
struct homogeneous
  : category
{
    typedef homogeneous type; // self-returning, for convenience
};

}}}} // namespace boost::sequence::algorithm::fixed_size

#endif // BOOST_SEQUENCE_ALGORITHM_FIXED_SIZE_CATEGORY_DWA200559_HPP
