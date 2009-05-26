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

#ifndef MTL_PAR_RANK_OSTREAM_INCLUDE
#define MTL_PAR_RANK_OSTREAM_INCLUDE

#include <boost/static_assert.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/bool.hpp>

#include <boost/numeric/mtl/operation/parallel_utilities.hpp>

namespace mtl { namespace par {


/// ostream that writes on each processor ; by default on std::cout using MPI_WORLD
/** Must not be used for distributed types! 
    If the template parameter \p PrintRank is true (the default) then the MPI rank is prepended.
    Note that this is done only once per expression because the first operator<< returns
    multiple_ostream with PrintRank set to false.
    The template parameter \p Serialize defines wether or not the output is serialized printing
    first on rank 0 of comm, then on rank 1 until rank n-1.
    With such ostream every output expression becomes a collective operation on communicator \p comm
    and must be called by EVERY process!  Otherwise you will experience a dead-lock.
    The last template argument handles the synchronization internally. 
    Do not create objects with \p Template set to true!
**/
template <bool PrintRank = true, bool Serialize = true, bool Temporary = false>
struct multiple_ostream
{
    /// Constructor for out or std::cout and MPI_WORLD
    multiple_ostream(std::ostream& out = std::cout) : out(out), comm(boost::mpi::communicator()) {} 
    /// Constructor for out and given communicator
    multiple_ostream(std::ostream& out, const boost::mpi::communicator& comm) : out(out), comm(comm) {} 
    /// Constructor for out and dist's communicator
    multiple_ostream(std::ostream& out, const base_distribution& dist) : out(out), comm(communicator(dist)) {} 

    ~multiple_ostream()
    {
	if (Serialize && Temporary) {
	    out.flush(); start_next(comm);
	}
    }

    typedef typename boost::mpl::if_c<Temporary, multiple_ostream&, multiple_ostream<false, Serialize, true> >::type return_type;

    template <typename T>
    return_type shift(const T& v, boost::mpl::true_)
    {
	out << v;
	return *this;
    }

    template <typename T>
    return_type shift(const T& v, boost::mpl::false_)
    {
	BOOST_STATIC_ASSERT((!traits::is_distributed<T>::value));
	if (Serialize)
	    wait_for_previous(comm);
	if (PrintRank)
	    out << comm.rank() << ": ";
	out << v;
	return return_type(out, comm);
    }

    template <typename T>
    return_type operator<<(const T& v)
    {
	return shift(v, boost::mpl::bool_<Temporary>());
    }

#if 0    
    // Output on temporary returns a reference (not an object)
    template <typename T>
    boost::enable_if_c<Temporary, multiple_ostream&>
    operator<<(const T& v)
    {
	out << v;
	return *this;
    }

    /// The output command
    template <typename T>
    boost::disable_if_c<Temporary, multiple_ostream<false, Serialize, true> >
    operator<<(const T& v)
    {
	BOOST_STATIC_ASSERT((!traits::is_distributed<T>::value));
	if (Serialize)
	    wait_for_previous(comm);
	if (PrintRank)
	    out << comm.rank() << ": ";
	out << v;
	return multiple_ostream<false, Serialize, false>(out, comm);
    }
#endif

    /// Flush output
    void flush() { out.flush(); }
private:
    std::ostream&            out;
    boost::mpi::communicator comm;
};


/// Serialized ostream that prepends rank in each expression, must be called in every process, for details see multiple_ostream.
typedef multiple_ostream<true, true, false> rank_ostream;

/// Non-serialized ostream that prepends rank in each expression, for details see multiple_ostream.
typedef multiple_ostream<true, false, false> nosync_rank_ostream;


}} // namespace mtl::par

#endif // MTL_PAR_RANK_OSTREAM_INCLUDE
