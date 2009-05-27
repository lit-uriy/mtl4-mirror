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

#ifndef MTL_PRINT_VECTOR_INCLUDE
#define MTL_PRINT_VECTOR_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/operation/local.hpp>
#include <boost/numeric/mtl/operation/communicator.hpp>
#include <boost/numeric/mtl/operation/parallel_utilities.hpp>

#ifdef MTL_HAS_MPI
#  include <boost/mpi.hpp>
#endif

namespace mtl { namespace vector {

template <typename Vector>
std::ostream& print_vector(Vector const& vector, std::ostream& out, int width, int precision, tag::universe)
{
    out << '{' << size(vector) 
	<< (traits::is_row_major< typename OrientedCollection<Vector>::orientation >::value ? "R" : "C") 
	<< "}[" ;
    for (size_t r = 0; r < size(vector); ++r) {
	out.fill (' '); 
	if (width)
	    out.width (width); 
	// out.flags (std::ios_base::right);
	if (precision)
	    out.precision(precision); 
	out << vector[r] << (r < size(vector) - 1 ? "," : "");
    }
    return out << "]";
}

#ifdef MTL_HAS_MPI
template <typename Vector>
inline std::ostream& print_vector(Vector const& vector, std::ostream& out, int width, int precision, tag::distributed)
{
    const boost::mpi::communicator& comm(communicator(vector));
    wait_for_previous(comm);
    if (comm.rank()) out << "||";
    print_vector(local(vector), out, width, precision);
    out.flush();
    start_next(comm);
}
#endif

template <typename Vector>
inline std::ostream& print_vector(Vector const& vector, std::ostream& out= std::cout, int width= 0, int precision= 0)
{
    return print_vector(vector, out, width, precision, typename mtl::traits::category<Vector>::type());
}





}} // namespace mtl::vector

#endif // MTL_PRINT_VECTOR_INCLUDE
