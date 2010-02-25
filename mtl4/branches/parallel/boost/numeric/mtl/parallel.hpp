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

#ifndef MTL_PARALLEL_INCLUDE
#define MTL_PARALLEL_INCLUDE

#include <boost/numeric/mtl/vector/distributed.hpp>
#include <boost/numeric/mtl/matrix/distributed.hpp> 

#include <boost/numeric/mtl/par/comm_scheme.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/par/mpi_log.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>

#include <boost/numeric/mtl/operation/parallel_utilities.hpp>

#include <boost/numeric/mtl/par/agglomerate.hpp>
#include <boost/numeric/mtl/par/global_columns.hpp>
#include <boost/numeric/mtl/par/new_global_map.hpp>
#include <boost/numeric/mtl/par/global_non_zeros.hpp>
#include <boost/numeric/mtl/par/dist_mat_cvec_mult.hpp>
#include <boost/numeric/mtl/par/migrate_matrix.hpp>
#include <boost/numeric/mtl/par/migrate_vector.hpp>

#include <boost/numeric/mtl/interface/parmetis_migration.hpp>
#include <boost/numeric/mtl/interface/parmetis.hpp>

#endif // MTL_PARALLEL_INCLUDE
