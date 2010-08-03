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

#ifdef MTL_HAS_MPI

#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>

#include <boost/mpl/bool.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

namespace mtl { 

    namespace par {

	/// Base class for all distributions
	class base_distribution
	{
	  public:
	    typedef std::size_t     size_type;
	    
	    explicit base_distribution (const boost::mpi::communicator& comm= boost::mpi::communicator()) 
	      : comm(comm), my_rank(comm.rank()), my_size(comm.size()) {}
	    
	    /// Distributions not specified further or of different types are considered different
	    bool operator==(const base_distribution&) const { return false; }
	    /// Distributions not specified further or of different types are considered different
	    bool operator!=(const base_distribution& d) const { return true; }

	    /// Current communicator
	    friend inline const boost::mpi::communicator& communicator(const base_distribution& d);

	    int rank() const { return my_rank; }
	    int size() const { return my_size; }

	    friend inline std::ostream& operator<< (std::ostream& out, const base_distribution& d)
	    {
		return out << "Basic distribution of size " << d.my_size; 
	    }

	  protected:
	    boost::mpi::communicator comm;
	    int               my_rank, my_size;
	};

        inline const boost::mpi::communicator& communicator(const base_distribution& d) { return d.comm; }
	
	/// Block distribution
	class block_distribution 
	  : public base_distribution
	{
	    void init(size_type n)
	    {
		size_type procs= my_size, inc= n / procs, mod= n % procs;
		starts[0]= 0;
		for (size_type i= 0; i < procs; ++i)
		    starts[i+1]= starts[i] + inc + (i < mod);
		assert(starts[procs] == n);
	    }

	  public:
	    /// Distribution for n (global) entries
	    explicit block_distribution(size_type n, const boost::mpi::communicator& comm= boost::mpi::communicator())
	      : base_distribution(comm), starts(comm.size()+1)
	    { init(n); }

	    /// Distribution vector
	    explicit block_distribution(const std::vector<size_type>& starts, 
					const boost::mpi::communicator& comm= boost::mpi::communicator())
	      : base_distribution(comm), starts(starts)
	    {}

	    // should be generated
	    // block_distribution(const block_distribution& src) : base_distribution(src), starts(src.starts) {}

	    /// Change number of global entries to n
	    void resize(size_type n) { init(n); }

	    /// Change number of global entries to n
	    /** If all entries are in first partition (e.g. agglomeration) this one is enlarged
		otherwise the last partition. **/
	    void stretch(size_type n) 
	    { 
		assert(my_size > 0);
		if (n <= starts[my_size])  // large enough
		    return;
		if (my_size > 1 && starts[1] == starts[my_size])
		    for (size_type i= 1; i < size_type(my_size); ++i)
			starts[i]= n;
		else
		    starts[my_size]= n;
	    }

	    /// Set up from a vector with the size of each partition (performs partial sum)
	    void setup_from_local_sizes(const std::vector<size_type>& lsizes)
	    {
		starts.resize(lsizes.size() + 1);
		starts[0]= 0;
		for (std::size_t i= 0; i < lsizes.size(); i++)
		    starts[i+1]= starts[i] + lsizes[i];
	    }

	    /// Two block distributions are equal if they have the same blocks and same communicator
	    bool operator==(const block_distribution& dist) const { return comm == dist.comm && starts == dist.starts; }
	    /// Two block distributions are different if they have the different blocks or communicators
	    bool operator!=(const block_distribution& dist) const { return !(*this == dist); }

	    /// For n global entries, how many are on processor p?
	    template <typename Size>
	    Size num_local(Size n, int p) const
	    { 
		return std::max(std::min(n, starts[p+1]), starts[p]) - starts[p]; 
	    }
	    
	    /// For n global entries, how many are on my processor?
	    template <typename Size>
	    Size num_local(Size n) const { return num_local(n, my_rank); }

	    /// The maximal number of global entries 
	    size_type max_global() const { return starts[my_size]; }

	    /// Is the global index \p n on my processor
	    bool is_local(size_type n) const { return n >= starts[my_rank] && n < starts[my_rank+1]; }

	    /// Global index of local index \p n on rank \p p
	    /** Fails with empty partitions, consider start_index. **/
	    template <typename Size>
	    Size local_to_global(Size n, int p) const
	    {
		//MTL_DEBUG_THROW_IF(n >= starts[p+1] - starts[p], range_error);
		return n + starts[p];
	    }

	    /// Global index of local index \p n on my processor
	    /** Fails with empty partitions, consider start_index. **/
	    template <typename Size>
	    Size local_to_global(Size n) const { return local_to_global(n, my_rank); }

	    /// Local index of \p n under the condition it is on rank \p p
	    template <typename Size>
	    Size global_to_local(Size n, int p) const
	    {
		//MTL_DEBUG_THROW_IF(n < starts[p] || n >= starts[p+1], range_error);
		return n - starts[p];
	    }

	    /// Local index of \p n under the condition it is on my processor
	    template <typename Size>
	    Size global_to_local(Size n) const { return global_to_local(n, my_rank); }

	    /// On which rank is global index n?
	    int on_rank(size_type n) const 
	    { 
		if (n < starts[0] || n >= starts[my_size]) std::cerr << "out of range with n == " << n << " and starts == " << starts << std::endl;
		MTL_DEBUG_THROW_IF(n < starts[0] || n >= starts[my_size], range_error());
		std::vector<size_type>::const_iterator lbound( std::lower_bound(starts.begin(), starts.end(), n));
		return lbound - starts.begin() - int(*lbound != n);
	    }

	    friend inline std::ostream& operator<< (std::ostream& out, const block_distribution& d)
	    {
		out << "Block distribution: ";
#             ifndef MTL_DISABLE_STD_OUTPUT_OPERATOR
		out << d.starts; 
#             endif
		return out;
	    }

	    //private:
	    // No default constructor
	    block_distribution() {}

	    /// Lowest global index of each block and total size
	    std::vector<size_type> starts;
	};

	/// Cyclic distribution
	// Not tested yet
	class cyclic_distribution : public base_distribution
	{
	  public:
	    /// Construction of cyclic distribution 
	    explicit cyclic_distribution(const boost::mpi::communicator& comm= boost::mpi::communicator()) 
	      : base_distribution(comm) {}

	    /// Construction of cyclic distribution of size \p n with communicator \p comm
	    /** Size \p n is ignored and the constructor only exist for compatibility with other distributions. **/
	    explicit cyclic_distribution(size_type, const boost::mpi::communicator& comm= boost::mpi::communicator()) 
	      : base_distribution(comm) {}

	    /// Change number of global entries to n (only dummy)
	    void resize(size_type) {}
	    /// Change number of global entries to n (only dummy)
	    void stretch(size_type) {}

	    /// Two cyclic distributions are equal if they have the same communicator size
	    bool operator==(const cyclic_distribution& dist) const { return size() == dist.size(); }
	    /// Two cyclic distributions are different if they have the different communicator sizes
	    bool operator!=(const cyclic_distribution& dist) const { return !(*this == dist); }

	    /// For n global entries, how many are on processor p?
	    template <typename Size>
	    Size num_local(Size n, int p) const
	    { 
		return n / my_size + (my_rank < int(n) % my_size);
	    }
	    
	    /// For n global entries, how many are on my processor?
	    template <typename Size>
	    Size num_local(Size n) const { return num_local(n, my_rank); }

	    /// The maximal number of global entries (unlimited for cyclic)
	    size_type max_global() const { return std::numeric_limits<size_type>::max(); }

	    /// Is the global index \p n on my processor
	    bool is_local(size_type n) const { return n % my_size == size_type(my_rank); }

	    /// Global index of local index \p n on rank \p p
	    template <typename Size>
	    Size local_to_global(Size n, int p) const {	return n * my_size + p; }

	    /// Global index of local index \p n on my processor
	    template <typename Size>
	    Size local_to_global(Size n) const { return local_to_global(n, my_rank); }

	    /// Local index of \p n under the condition it is on rank \p p
	    template <typename Size>
	    Size global_to_local(Size n, int p) const
	    {
		//MTL_DEBUG_THROW_IF(n % my_size != p, range_error);
		return n / my_size;
	    }

	    /// Local index of \p n under the condition it is on my processor
	    template <typename Size>
	    Size global_to_local(Size n) const { return global_to_local(n, my_rank); }

	    /// On which rank is global index n?
	    int on_rank(size_type n) const { return n % my_size; }

	    friend inline std::ostream& operator<< (std::ostream& out, const cyclic_distribution& d)
	    {
		return out << "Cyclic distribution with cycle size " << d.my_size; 
	    }
	};

	/// Block cyclic distribution
	// Not tested yet
	class block_cyclic_distribution : public base_distribution
	{
	  public:
	    /// Construction of block cyclic distribution 
	    /** Interface is not consistent with other distribution where first size parameter specifies the size of the collection **/
	    explicit block_cyclic_distribution(size_type bsize, const boost::mpi::communicator& comm= boost::mpi::communicator()) 
	      : base_distribution(comm), bsize(bsize), sb(bsize * my_size) {}
	    
	    /// Change number of global entries to n (only dummy)
	    void resize(size_type) {}
	    /// Change number of global entries to n (only dummy)
	    void stretch(size_type) {}

	    /// Two block cyclic distributions are equal if they have the same communicator
	    bool operator==(const block_cyclic_distribution& dist) const { return size() == dist.size() && bsize == dist.bsize; }
	    /// Two block cyclic distributions are different if they have the different communicators
	    bool operator!=(const block_cyclic_distribution& dist) const { return !(*this == dist); }

	    /// For n global entries, how many are on processor p?
	    template <typename Size>
	    Size num_local(Size n, int p) const
	    { 
		Size full_blocks(n / sb), in_full_blocks(n % sb), my_block(p * bsize);
		return full_blocks * bsize + std::max(0ul, std::min(in_full_blocks - my_block, bsize));
	    }
	    
	    /// For n global entries, how many are on my processor?
	    template <typename Size>
	    Size num_local(Size n) const { return num_local(n, my_rank); }

	    /// The maximal number of global entries (unlimited for block cyclic)
	    size_type max_global() const { return std::numeric_limits<size_type>::max(); }

	    /// Is the global index \p n on my processor
	    bool is_local(size_type n) const { return on_rank(n) == my_rank; }

	    /// Global index of local index \p n on rank \p p
	    template <typename Size>
	    Size local_to_global(Size n, int p) const 
	    {	
		return (n / bsize) * bsize * my_size + my_rank * bsize + (n % bsize);
	    }

	    /// Global index of local index \p n on my processor
	    template <typename Size>
	    Size local_to_global(Size n) const { return local_to_global(n, my_rank); }

	    /// Local index of \p n under the condition it is on rank \p p
	    template <typename Size>
	    Size global_to_local(Size n, int p) const
	    {
		MTL_DEBUG_THROW_IF(on_rank(n) != p, range_error());
		return num_local(n, p);
	    }

	    /// Local index of \p n under the condition it is on my processor
	    template <typename Size>
	    Size global_to_local(Size n) const { return global_to_local(n, my_rank); }

	    /// On which rank is global index n?
	    int on_rank(size_type n) const { return (n % sb) / bsize; }

	    friend inline std::ostream& operator<< (std::ostream& out, const block_cyclic_distribution& d)
	    {
		return out << "Cyclic distribution with cycle size " << d.my_size << " and block size " << d.bsize; 
	    }

	  private:
	    size_type bsize, sb;
	};

	/// Replicated storage of data where every proc has a copy of the considered object and is supposed to the same operations on it.
	class replication
	  : public base_distribution
	{
	  public:
	    /// Replicated distribution with communicator \p comm
	    replication(const boost::mpi::communicator& comm= boost::mpi::communicator())
	      : base_distribution(comm) {}

	    /// Replicated distribution of size \p n with communicator \p comm
	    /** Size \p n is ignored and the constructor only exist for compatibility with other distributions. **/
	    replication(size_type, const boost::mpi::communicator& comm= boost::mpi::communicator())
	      : base_distribution(comm) {}

	    /// Change number of global entries to n (only dummy)
	    void resize(size_type) {}
	    /// Change number of global entries to n (only dummy)
	    void stretch(size_type) {}

	    /// Two replications are equal if they have the same communicator size
	    bool operator==(const cyclic_distribution& dist) const { return size() == dist.size(); }
	    /// Two replications are different if they have the different communicator sizes
	    bool operator!=(const cyclic_distribution& dist) const { return !(*this == dist); }

	    /// For \p n global entries, how many are on processor \p p? The answer is n.
	    template <typename Size>
	    Size num_local(Size n, int) const { return n; }

	    /// For n global entries, how many are on my processor? The answer is n.
	    template <typename Size>
	    Size num_local(Size n) const { return n; }

	    /// The maximal number of global entries (unlimited for replications)
	    size_type max_global() const { return std::numeric_limits<size_type>::max(); }

	    /// Is the global index \p n on my processor? The answer is always yes.
	    bool is_local(size_type n) const { return true; }

	    /// Global index of local index \p n on rank \p p
	    template <typename Size>
	    Size local_to_global(Size n, int p) const {	return n; }

	    /// Global index of local index \p n on my processor
	    template <typename Size>
	    Size local_to_global(Size n) const { return n; }

	    /// Local index of \p n under the condition it is on rank \p p
	    template <typename Size>
	    Size global_to_local(Size n, int p) const { return n; }

	    /// Local index of \p n under the condition it is on my processor
	    template <typename Size>
	    Size global_to_local(Size n) const { return n; }

	    /// On which rank is global index n? Returns own rank on each processor.
	    int on_rank(size_type n) const { return my_rank; }

	    friend inline std::ostream& operator<< (std::ostream& out, const replication& d)
	    {
		return out << "Replication";
	    }
	};


	/// Vectorized version of local_to_global with respect to \p dist
	template <typename Dist>
	void inline local_to_global(const Dist& dist, std::vector<base_distribution::size_type>& indices, int p)
	{
	    for (std::size_t i= 0; i < indices.size(); i++)
		indices[i]= dist.local_to_global(indices[i], p);
	}
	
    } // namespace par

    namespace traits {

	/// Type trait to check for block distribution
	template <typename Dist> struct is_block_distribution : boost::mpl::false_ {};
	template <> struct is_block_distribution<mtl::par::block_distribution> : boost::mpl::true_ {};

    }// traits

} // namespace mtl

#endif // MTL_HAS_MPI

#endif // MTL_DISTRIBUTION_INCLUDE
