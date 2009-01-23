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

#include <vector>
#include <algorithm>

#include <boost/mpl/bool.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>

namespace mtl { 

    namespace tag {

	
	struct distributed {};

	struct block_distributed : distributed {};
	
	struct concentrated {};

    } // namespace tag


    namespace par {

	namespace mpi = boost::mpi;

	/// Base class for all distributions
	class distribution
	{
	public:
	    typedef std::size_t     size_type;
	    
	    explicit distribution (const mpi::communicator& comm= mpi::communicator()) 
		: comm(comm), my_rank(comm.rank()), my_size(comm.size()) {}
	    
	    /// Distributions not specified further or of different types are considered different
	    bool operator==(const distribution&) const { return false; }
	    /// Distributions not specified further or of different types are considered different
	    bool operator!=(const distribution& d) const { return true; }

	    /// Current communicator
	    mpi::communicator communicator() const { return comm; }

	    int rank() const { return my_rank; }
	    int size() const { return my_size; }
	protected:
	    mpi::communicator comm;
	    int               my_rank, my_size;
	};

	
	/// Block row distribution
	class block_distribution : public distribution
	{
	    void init(size_type n)
	    {
		size_type procs= comm.size(), inc= n / procs, mod= n % procs;
		starts[0]= 0;
		for (size_type i= 0; i < procs; ++i)
		    starts[i+1]= starts[i] + inc + (i < mod);
		assert(starts[procs] == n);
	    }

	public:
	    /// Distribution for n (global) rows
	    explicit block_distribution(size_type n, const mpi::communicator& comm= mpi::communicator())
		: distribution(comm), starts(comm.size()+1)
	    { init(n); }

#if 0
	    /// For genericity construct from # of global rows and columns
	    explicit block_distribution(size_type grows, size_type gcols, 
					const mpi::communicator& comm= mpi::communicator())
		: distribution(comm), starts(comm.size()+1)
	    { init(grows); }
#endif

	    /// Distribution vector
	    explicit block_distribution(const std::vector<size_type>& starts, 
					const mpi::communicator& comm= mpi::communicator())
		: distribution(comm), starts(starts)
	    {}

	    bool operator==(const block_distribution& dist) const { return starts == dist.starts; }
	    bool operator!=(const block_distribution& dist) const { return starts != dist.starts; }

	    /// For n global entries, how many are on processor p?
	    template <typename Size>
	    Size num_local(Size n, int p) const
	    { 
		return std::max(std::min(n, starts[p+1]), starts[p]) - starts[p]; 
	    }
	    
	    /// For n global entries, how many are on my processor?
	    template <typename Size>
	    Size num_local(Size n) const { return num_local(n, my_rank); }


	    bool is_local(size_type n) const { return n >= starts[my_rank] && n < starts[my_rank+1]; }

	    /// Global index of local index \p n on rank \p p
	    template <typename Size>
	    Size local_to_global(Size n, int p) const
	    {
		MTL_DEBUG_THROW_IF(n >= starts[p+1] - starts[p], range_error);
		return n + starts[p];
	    }

	    /// Global index of local index \p n on my processor
	    template <typename Size>
	    Size local_to_global(Size n) const { return local_to_global(n, my_rank); }

	    /// Local index of \p n under the condition it is on rank \p p
	    template <typename Size>
	    Size global_to_local(Size n, int p) const
	    {
		MTL_DEBUG_THROW_IF(n < starts[p] || n >= starts[p+1], range_error);
		return n - starts[p];
	    }

	    /// Local index of \p n under the condition it is on my processor
	    template <typename Size>
	    Size global_to_local(Size n) const { return global_to_local(n, my_rank); }

	    /// On which rank is global index n?
	    int on_rank(size_type n) const 
	    { 
		if (n < starts[0] || n >= starts[my_size])
		    std::cerr << "out of range with n == " << n << "max == " << starts[my_size] << std::endl;
		MTL_DEBUG_THROW_IF(n < starts[0] || n >= starts[my_size], range_error);
		std::vector<size_type>::const_iterator lbound( std::lower_bound(starts.begin(), starts.end(), n));
		return lbound - starts.begin() - int(*lbound != n);
	    }

	private:
	    /// No default constructor
	    block_distribution() {}

	    /// Lowest global index of each block and total size
	    std::vector<size_type> starts;
	};

    }


    namespace traits {

	template <typename T>
	struct is_distributed : boost::mpl::false_ {}; 

	template <typename Matrix, typename Distribution, typename DistributionFrom>
	struct is_distributed<mtl::matrix::distributed<Matrix, Distribution, DistributionFrom> > : boost::mpl::true_ {};

	template <typename Vector, typename Distribution>
	struct is_distributed<mtl::vector::distributed<Vector, Distribution> > : boost::mpl::true_ {};

    }


} // namespace mtl

#endif // MTL_DISTRIBUTION_INCLUDE
