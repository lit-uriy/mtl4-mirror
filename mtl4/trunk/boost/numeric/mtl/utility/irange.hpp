// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_IRANGE_INCLUDE
#define MTL_IRANGE_INCLUDE

#include <limits>


namespace mtl {

/// Class to define a half open index ranges 
class irange
{
    irange() {} // no default constructor

  public:

    typedef std::size_t size_type;

    /// Create an index range of [start, finish)
    explicit irange(size_type start, size_type finish) : my_start(start), my_finish(finish) {}

    /// Create an index range of [0, finish)
    explicit irange(size_type finish) : my_start(0), my_finish(finish) {}

    /// First index in range
    size_type start() { return my_start; } 
    /// Past-end index in range
    size_type finish() { return my_finish; }
    /// Number of indices
    size_type size() { return my_finish - my_start; }

  private:

    size_type my_start, my_finish;
};

///
const std::size_t imax= std::numeric_limits<std::size_t>::max();

} // namespace mtl

#endif // MTL_IRANGE_INCLUDE
