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

#ifndef MTL_DISTRIBUTION_INCLUDE
#define MTL_DISTRIBUTION_INCLUDE

#include <boost/mpl/bool.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl { 

    namespace tag {

	
	struct distributed {};

	struct block_distributed : distributed {};
	
	struct row_block_distributed : block_distributed {};

	struct non_distributed {};

    } // namespace tag


    namespace par {

	namespace mpi = boost::mpi;

	/// Base class for all distributions
	struct distribution
	{
	    /// Distributions not specified further or of different types are considered different
	    bool operator==(const distribution&) const { return false; }
	    virtual bool operator!=(const distribution& d) const { return !(*this == d); }

	    mpi::communicator communicator() const { return comm; }
	protected:
	    mpi::communicator comm;
	};

	
	/// Base class for all row distributions
	struct row_distribution : public distribution {};

	/// Block row distribution
	struct block_row_distribution : public row_distribution
	{
	private:
	    void init(std::size_t n)
	    {
		std::size_t procs= comm.size(), inc= n / procs, mod= n % procs;
		starts[0]= 0;
		for (std::size_t i= 0; i < procs; ++i)
		    starts[i+1]= starts[i] + inc + (i < mod);
		assert(starts[procs] == n);
	    }

	public:
	    /// Distribution for n (global) rows
	    explicit block_row_distribution(std::size_t n, const mpi::communicator& comm= mpi::communicator())
		: comm(comm), starts(comm.size()+1)
	    { init(n); }

	    /// For genericity construct from # of global rows and columns
	    explicit block_row_distribution(std::size_t grows, std::size_t gcols, const mpi::communicator& comm= mpi::communicator())
		: comm(comm), starts(comm.size()+1)
	    { init(grows); }
	    

	    /// Distribution vector
	    explicit block_row_distribution(const std::vector<std::size_t>& starts, const mpi::communicator& comm= mpi::communicator())
		: comm(comm), starts(starts)
	    {}

	    bool operator==(const block_row_distribution& dist) const { return starts == dist.starts; }

	private:
	    /// No default constructor
	    block_row_distribution() {}

	    /// Lowest global index of each block and total size
	    std::vector<std::size_t> starts;
	};

    }



    namespace traits {

	template <typename T>
	struct is_distributed : boost::mpl::false_ {}; 

	template <typename Matrix, typename Distribution>
	struct is_distributed<mtl::matrix::distributed<Matrix, Distribution> > : boost::mpl::true_ {};

	template <typename Vector, typename Distribution>
	struct is_distributed<mtl::vector::distributed<Vector, Distribution> > : boost::mpl::true_ {};

    }


} // namespace mtl

#endif // MTL_DISTRIBUTION_INCLUDE
