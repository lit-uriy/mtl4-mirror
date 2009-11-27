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

#ifndef MTL_PAR_NEW_GLOBAL_MAP_INCLUDE
#define MTL_PAR_NEW_GLOBAL_MAP_INCLUDE

#include <map>
#include <utility>
#include <vector>
#include <boost/numeric/mtl/par/distribution.hpp>

#include <boost/numeric/mtl/operation/std_output_operator.hpp>
#include <boost/numeric/mtl/par/rank_ostream.hpp>
#include <boost/numeric/mtl/par/single_ostream.hpp>

namespace mtl { namespace par {


/// Build a map that maps global indices in the old distribution to those in the new one 
/** Remark: the table in migration object can only map indices in the local range, others are stored
    on other processors and need communication. **/
template <typename Indices, typename Map>
void new_global_map(const block_migration& migration, const Indices& indices, Map& map)
{
    using std::size_t; using std::pair; using std::vector;
    typedef typename Indices::value_type                   value_type;
    typedef vector<vector<value_type> >                    buffer_type;
    typedef typename vector<value_type>::const_iterator    iter_type;
    typedef vector<vector<pair<value_type, value_type> > > new_buffer_type;
    typedef typename vector<pair<value_type, value_type> >::const_iterator       new_iter_type;

    mtl::par::multiple_ostream<> mout;

    const block_distribution old_dist(migration.old_dist);
    buffer_type send_buffers(old_dist.size()), recv_buffers;

    // if local index put into map otherwise ask according processor
    for (size_t i= 0; i < indices.size(); ++i) {
	value_type ind= indices[i];
	if (old_dist.is_local(ind))
	    map[ind]= migration.new_global(old_dist.global_to_local(ind));
	else
	    send_buffers[old_dist.on_rank(ind)].push_back(ind);
    }
    all_to_all(communicator(old_dist), send_buffers, recv_buffers);

    // tell asking processor where are all indices gone
    new_buffer_type new_send_buffers(old_dist.size()), new_recv_buffers;
    for (size_t p= 0; p < recv_buffers.size(); p++)
	for (iter_type it= recv_buffers[p].begin(), end= recv_buffers[p].end(); it != end; ++it)
	    new_send_buffers[p].push_back(std::make_pair(*it, migration.new_global(old_dist.global_to_local(*it))));
    all_to_all(communicator(old_dist), new_send_buffers, new_recv_buffers);

    // put index mapping I got from others into map
    for (size_t p= 0; p < new_recv_buffers.size(); p++)
	for (new_iter_type it= new_recv_buffers[p].begin(), end= new_recv_buffers[p].end(); it != end; ++it)
	    map[it->first]= it->second;
}


}} // namespace mtl::par

#endif // MTL_PAR_NEW_GLOBAL_MAP_INCLUDE
