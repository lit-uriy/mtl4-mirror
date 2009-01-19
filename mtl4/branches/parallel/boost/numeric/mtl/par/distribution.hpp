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
	
	struct row_block_distributed : block_distributed {};

	struct non_distributed {};

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

	
	/// Base class for all row distributions
	class row_distribution : public distribution 
	{
	public:
	    explicit row_distribution(const mpi::communicator& comm= mpi::communicator()) : distribution(comm) {}
	};

	/// Block row distribution
	class block_row_distribution : public row_distribution
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
	    explicit block_row_distribution(size_type n, 
					    const mpi::communicator& comm= mpi::communicator())
		: row_distribution(comm), starts(comm.size()+1)
	    { init(n); }

	    /// For genericity construct from # of global rows and columns
	    explicit block_row_distribution(size_type grows, size_type gcols, 
					    const mpi::communicator& comm= mpi::communicator())
		: row_distribution(comm), starts(comm.size()+1)
	    { init(grows); }
	    

	    /// Distribution vector
	    explicit block_row_distribution(const std::vector<size_type>& starts, 
					    const mpi::communicator& comm= mpi::communicator())
		: row_distribution(comm), starts(starts)
	    {}

	    bool operator==(const block_row_distribution& dist) const { return starts == dist.starts; }

	    size_type local_num_rows(size_type gr) const
	    { 
		return std::max(std::min(gr, starts[my_rank+1]), starts[my_rank]) - starts[my_rank]; 
	    }
	    
	    size_type local_num_cols(size_type gc) const { return gc; }

	    bool is_local(size_type gr, size_type gc) const { return gr >= starts[my_rank] && gr < starts[my_rank+1]; }

	    template <typename Size>
	    Size local_row(Size gr) const
	    {
		MTL_DEBUG_THROW_IF(gr < starts[my_rank] || gr >= starts[my_rank+1], range_error);
		return gr - starts[my_rank];
	    }

	    template <typename Size>
	    Size local_col(Size gc) const { return gc; }

	    int on_rank(size_type gr, size_type gc) const 
	    { 
		MTL_DEBUG_THROW_IF(gr < starts[0] || gr >= starts[my_size], range_error);
		std::vector<size_type>::const_iterator lbound( std::lower_bound(starts.begin(), starts.end(), gr));
		//const size_type *lbound;
		//lbound= std::lower_bound(starts.begin(), starts.end(), gr);
		return lbound - starts.begin() - int(*lbound != gr);
	    }
	private:
	    /// No default constructor
	    block_row_distribution() {}

	    /// Lowest global index of each block and total size
	    std::vector<size_type> starts;
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
