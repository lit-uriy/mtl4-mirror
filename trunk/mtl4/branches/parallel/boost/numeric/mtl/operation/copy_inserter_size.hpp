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

#ifndef MTL_DETAIL_COPY_INSERTER_SIZE_INCLUDE
#define MTL_DETAIL_COPY_INSERTER_SIZE_INCLUDE

namespace mtl { namespace detail {

	// Adapt inserter size to operation
	template <typename Updater> struct copy_inserter_size {};
	
	// Specialization for store
	template <typename Value>
	struct copy_inserter_size< operations::update_store<Value> >
	{
	    template <typename MatrixSrc, typename MatrixDest>
	    static inline int apply(const MatrixSrc& src, const MatrixDest& dest)
	    {
		return int(src.nnz() * 1.2 / dest.dim1());
	    }
	};

	struct sum_of_sizes
	{
	    template <typename MatrixSrc, typename MatrixDest>
	    static inline int apply(const MatrixSrc& src, const MatrixDest& dest)
	    {	return int((src.nnz() + dest.nnz()) * 1.2 / dest.dim1()); }
	};
	    	
	// Specialization for plus and minus
	template <typename Value> struct copy_inserter_size< operations::update_plus<Value> > : sum_of_sizes {};
	template <typename Value> struct copy_inserter_size< operations::update_minus<Value> > : sum_of_sizes {};


}} // namespace mtl::detail

#endif // MTL_DETAIL_COPY_INSERTER_SIZE_INCLUDE
