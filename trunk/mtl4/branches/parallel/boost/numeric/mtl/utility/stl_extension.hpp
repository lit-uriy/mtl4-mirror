// Software License for MTL
// 
// Copyright (c) 2007 The Trustees of Indiana University. 
//               2008 Dresden University of Technology and the Trustees of Indiana University. 
// All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_STL_EXTENSION_INCLUDE
#define MTL_STL_EXTENSION_INCLUDE

#include <algorithm>

namespace std {

/// Consume container \p consumed by appending its data to \consumer and emptying it
/** Works for std::vectors (general requirements pending). **/
template <typename Container>
void inline consume(Container& consumer, Container& consumed)
{
    consumer.insert(consumer.end(), consumed.begin(), consumed.end());
    Container tmp;
    swap(consumed, tmp);
}

/// Sort and (really) remove repeated entries 
/** Works for std::vectors (general requirements pending). **/
template <typename Container>
void inline only_unique(Container& c)
{
    sort(c.begin(), c.end());
    typename Container::iterator new_end = unique(c.begin(), c.end());
    c.erase(new_end, c.end());
}

} // namespace std

#endif // MTL_STL_EXTENSION_INCLUDE
