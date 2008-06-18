// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef MTL_DETAIL_ADJUST_CURSOR_INCLUDE
#define MTL_DETAIL_ADJUST_CURSOR_INCLUDE

namespace mtl { namespace detail {

    template <typename Size, typename Cursor>
    void inline adjust_cursor(Size diff, Cursor& c, tag::dense) { c+= diff; }
    
    template <typename Size, typename Cursor>
    void inline adjust_cursor(Size diff, Cursor& c, tag::sparse) {}
  

}} // namespace mtl::detail

#endif // MTL_DETAIL_ADJUST_CURSOR_INCLUDE
