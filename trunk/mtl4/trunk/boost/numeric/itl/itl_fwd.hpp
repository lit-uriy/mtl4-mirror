// Software License for MTL
// 
// Copyright (c) 2007-2008 The Trustees of Indiana University. All rights reserved.
// Authors: Peter Gottschling and Andrew Lumsdaine
// 
// This file is part of the Matrix Template Library
// 
// See also license.mtl.txt in the distribution.

#ifndef ITL_ITL_FWD_INCLUDE
#define ITL_ITL_FWD_INCLUDE

namespace itl {

    template <class Real> class basic_iteration;
    template <class Real> class noisy_iteration;

    template <typename Solver, typename VectorIn, bool trans> class solver_proxy;

    namespace pc {

	template <typename Matrix> class diagonal;

    } //  namespace pc

} // namespace itl

#endif // ITL_ITL_FWD_INCLUDE
