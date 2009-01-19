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

#ifndef MTL_MATRIX_DISTRIBUTED_INCLUDE
#define MTL_MATRIX_DISTRIBUTED_INCLUDE

#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>

namespace mtl { namespace matrix {


template <typename Matrix, typename Distribution = par::block_row_distribution>
class distributed
{
public:
    typedef typename Collection<Matrix>::size_type  size_type;
    typedef typename Collection<Matrix>::value_type value_type;
    typedef Distribution                            distribution_type;
    
    typedef Matrix                                  concentrated; // concentrated as antonym to distributed

    explicit distributed(size_type grows, size_type gcols 
			 // , const Distribution& dist= Distribution(grows, gcols)
			 ) 
	: dist(grows, gcols), local_matrix(dist.local_num_rows(grows), dist.local_num_cols(gcols)) {}

    explicit distributed(size_type grows, size_type gcols, 
			 const Distribution& dist) 
	: dist(dist), local_matrix(dist.local_num_rows(grows), dist.local_num_cols(gcols)) {}


    boost::mpi::communicator communicator() const { return dist.communicator(); }

    template <typename DistMatrix, typename Updater> friend class distributed_inserter;

    Distribution      dist;
    Matrix            local_matrix;
    Matrix*           remote_matrices;
};

template <typename DistributedMatrix, 
	  typename Updater = mtl::operations::update_store<typename Collection<DistributedMatrix>::value_type> >
class distributed_inserter
{
    typename DistributedMatrix::distribution_type const& dist() const { return distributed_matrix.dist; }
    int rank() const { return dist().rank(); }
    int size() const { return dist().size(); }

public:
    typedef distributed_inserter                           self;
    typedef DistributedMatrix                                     distributed_matrix_type;
    typedef typename Collection<DistributedMatrix>::size_type     size_type;
    typedef typename Collection<DistributedMatrix>::value_type    value_type;
    typedef typename DistributedCollection<DistributedMatrix>::concentrated  local_matrix_type;
    typedef local_matrix_type                                     matrix_type; // needed in functors
    typedef inserter<local_matrix_type, Updater>                  local_inserter_type;
    typedef operations::update_proxy<self, size_type>             proxy_type;
    typedef std::pair< std::pair<size_type, size_type>, value_type > entry_type;
    
    explicit distributed_inserter(DistributedMatrix& distributed_matrix, size_type slot_size = 5)
	: distributed_matrix(distributed_matrix), local_inserter(distributed_matrix.local_matrix, slot_size), 
	  send_buffers(distributed_matrix.dist.size()), recv_buffers(distributed_matrix.dist.size())
    {}

    ~distributed_inserter()
    {
	boost::mpi::all_to_all(dist().communicator(), send_buffers, recv_buffers);
	for (unsigned p= 0; p < size(); p++) {
	    const std::vector<entry_type>& my_buffer= recv_buffers[p];
	    for (unsigned i= 0; i < my_buffer.size(); i++) {
		const entry_type& entry= my_buffer[i];
		update(entry.first.first, entry.first.second, entry.second);
	    }}
    }

    operations::update_bracket_proxy<self, size_type> operator[] (size_type row)
    {
	return operations::update_bracket_proxy<self, size_type>(*this, row);
    }

    proxy_type operator() (size_type row, size_type col)
    {
	return proxy_type(*this, row, col);
    }

    template <typename Modifier>
    void modify(size_type row, size_type col, value_type val);

    void update(size_type row, size_type col, value_type val)
    {
	modify<Updater>(row, col, val);
    }

    template <typename Matrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_matrix_t<Matrix, Rows, Cols>& elements)
    {
	for (unsigned ri= 0; ri < elements.rows.size(); ri++)
	    for (unsigned ci= 0; ci < elements.cols.size(); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.matrix(ri, ci));
	return *this;
    }

    template <typename Matrix, typename Rows, typename Cols>
    self& operator<< (const matrix::element_array_t<Matrix, Rows, Cols>& elements)
    {
	for (unsigned ri= 0; ri < elements.rows.size(); ri++)
	    for (unsigned ci= 0; ci < elements.cols.size(); ci++)
		update (elements.rows[ri], elements.cols[ci], elements.array[ri][ci]);
	return *this;
    }

private:
    DistributedMatrix&                     distributed_matrix;
    local_inserter_type                    local_inserter;
    std::vector<std::vector<entry_type> >  send_buffers, recv_buffers;
};

template <typename DistributedMatrix, typename Updater>
template <typename Modifier>
inline void distributed_inserter<DistributedMatrix, Updater>::modify(size_type row, size_type col, value_type value)
{
    typename DistributedMatrix::distribution_type const& dist= distributed_matrix.dist;
    if (dist.is_local(row, col)) {
	size_type local_row= dist.local_row(row), local_col= dist.local_col(col);
	local_inserter.modify<Modifier>(local_row, local_col, value);
    } else
	send_buffers[dist.on_rank(row, col)].push_back(std::make_pair(std::make_pair(row, col), value));
}

}} // namespace mtl::matrix

#endif // MTL_MATRIX_DISTRIBUTED_INCLUDE
