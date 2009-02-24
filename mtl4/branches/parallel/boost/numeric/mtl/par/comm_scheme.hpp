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

#ifndef MTL_PAR_COMM_SCHEME_INCLUDE
#define MTL_PAR_COMM_SCHEME_INCLUDE

namespace mtl { namespace par {

//#define MTL_COMM_BLOCKING

/// Communication scheme
/** Can be choosen by defining macros. If no macros are set then 
    communication is performed in non-blocking, point-to-point, buffered manner.
    Certain communications may not support all eight combinations.
    Communication scheme can also be choosen explicitely in the function call. 
**/
struct comm_scheme :
#ifdef MTL_COMM_BLOCKING
	virtual tag::comm_blocking,
#else
	virtual tag::comm_non_blocking,
#endif
#ifdef MTL_COMM_COLLECTIVE
	virtual tag::comm_collective,
#else
	virtual tag::comm_p2p,
#endif
#ifdef MTL_COMM_DATA_TYPE
	virtual tag::comm_data_type
#else
	virtual tag::comm_buffer
#endif
{};


}} // namespace mtl::par

#endif // MTL_PAR_COMM_SCHEME_INCLUDE
