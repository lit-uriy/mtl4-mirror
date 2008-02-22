// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_CLONE_INCLUDE
#define MTL_CLONE_INCLUDE

namespace mtl {

/// Move-semantics-related anti-dot: always copy in constructor.
/** Some collections have referring semantics in copy constructors, e.g. sub-matrices.
    That means 
    \code
        Matrix B= sub_matrix(A, ...); 
    \endcode
    creates a sub-matrix of A in B. As a consequence, changes in B modify A and vice versa
    (unless it's outside the sub-matrix).
    In contrast, clone forces the copy semantics
    \code
        Matrix B= clone(sub_matrix(A, ...)); 
    \endcode
    B now contains the values of A's sub-matrix but is an independent matrix.
    Modifications to either A or B have no effect to each other.
**/
template <typename T>
inline T clone(const T& x) 
{ 
    return x; 
}


} // namespace mtl

#endif // MTL_CLONE_INCLUDE
