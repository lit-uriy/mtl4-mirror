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

#ifdef MTL_HAS_MPI

#include <iostream>
#include <utility>
#include <vector>
#include <map>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives/all_to_all.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/matrix/reorder.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/operation/parallel_utilities.hpp>
#include <boost/numeric/mtl/operation/for_each_nonzero.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>

namespace mtl { namespace matrix {


/// Distributed matrices
template <typename Matrix, typename RowDistribution, typename ColDistribution>
class distributed
{

public:
    typedef distributed                              self;
    typedef typename Collection<Matrix>::size_type   size_type;
    typedef typename Collection<Matrix>::value_type  value_type;
    typedef RowDistribution                          row_distribution_type;
    typedef ColDistribution                          col_distribution_type;
    
    typedef Matrix                                   local_type;
    typedef Matrix                                   remote_type; // Needs specialization
    typedef std::map<int, remote_type>               remote_map_type;
    typedef typename remote_map_type::const_iterator remote_map_const_iterator;

    /// Constructor for matrix with global size grows x gcols and default distribution.
    /** RowDistribution and ColDistribution must have same type. **/
    explicit distributed(size_type grows, size_type gcols) 
	: grows(grows), gcols(gcols), row_dist(grows), cdp(&this->row_dist), col_dist(*cdp),
	  local_matrix(row_dist.num_local(grows), col_dist.num_local(gcols))
    {}

    /// Constructor for matrix with global size grows x gcols and with given distribution.
    /** RowDistribution and ColDistribution must have same type. **/
    explicit distributed(size_type grows, size_type gcols, 
			 const RowDistribution& row_dist) 
	: grows(grows), gcols(gcols), row_dist(row_dist), cdp(&this->row_dist), col_dist(*cdp), 
	  local_matrix(row_dist.num_local(grows), col_dist.num_local(gcols))
    {}

    /// Constructor for matrix with global size grows x gcols and with different distributions for rows and columns.
    explicit distributed(size_type grows, size_type gcols, 
			 const RowDistribution& row_dist, const ColDistribution& col_dist) 
	: grows(grows), gcols(gcols), row_dist(row_dist), cdp(new ColDistribution(col_dist)), col_dist(*cdp), 
	  local_matrix(row_dist.num_local(grows), col_dist.num_local(gcols))
    {}

    ~distributed() { clean_cdp(); clean_remote_matrices(); }

    void clean_cdp() { if (cdp && cdp != &row_dist) delete cdp; }
    void clean_remote_matrices() { remote_matrices.clear(); }

    struct send_structure
    {
	send_structure(const dense_vector<size_type>& indices, size_type offset) : indices(indices), offset(offset) {}
	dense_vector<size_type>     indices;
	size_type                   offset;
    };

    struct recv_structure
    {
	recv_structure(size_type size, size_type offset) : size(size), offset(offset) {}
	size_type                   size, offset;
    };

    self& operator=(const self& src)
    {
	row_dist= src.row_dist;
	clean_cdp();
	col_dist_assign(src, boost::is_same<RowDistribution, ColDistribution>());
	local_matrix= src.local_matrix;
	// copy remote parts and such

	return *this;
    }

private:
    void col_dist_assign(const self& src, boost::mpl::true_)
    {
	// if dist and col_dist are the same object at source then col_dist is only a ref to dist
	if (&src.row_dist == src.cdp)
	    cdp= &row_dist;
	else
	    cdp= new ColDistribution(src.col_dist);	
    }

    void col_dist_assign(const self& src, boost::mpl::false_)
    {
	cdp= new ColDistribution(src.col_dist);	
    }


    friend inline const boost::mpi::communicator& communicator(const self& d) { return communicator(d.row_dist); }
			  
    
    template <typename DistMatrix, typename Updater> friend class distributed_inserter;

    friend inline size_type num_rows(const self& A) { return A.grows; }
    friend inline size_type num_cols(const self& A) { return A.gcols; }
    friend inline size_type size(const self& A) { return A.rows * A.gcols; }

    friend inline const local_type& local(const self& A) { return A.local_matrix; }
    friend inline local_type& local(self& A) { return A.local_matrix; }

    friend inline std::ostream& operator<< (std::ostream& out, const self& A) 
    {
	wait_for_previous(A.row_dist);
	const local_type& B= A.local_matrix;
	for (unsigned r= 0; r < num_rows(B); r++) {
	    for (int p= 0; p < A.col_dist.size(); p++) {
		out << '[';
		if (p == A.col_dist.rank())
		    for (unsigned c= 0; c < num_cols(B); c++)
			out << B[r][c] << (c < num_cols(B) - 1 ? " " : "");
		else {
		    remote_map_const_iterator it(A.remote_matrices.find(p));
		    if (it != A.remote_matrices.end()) {
			const remote_type& C= it->second; 
			for (unsigned c= 0; c < num_cols(C); c++)
			    out << C[r][c] << (c < num_cols(C) - 1 ? " " : "");
		    } else
			for (unsigned c= 0, nc= A.col_dist.num_local(num_cols(A), p); c < nc; c++)
			    out << '*' << (c < nc - 1 ? " " : "");
		}
		out << ']';
	    }
	    out << std::endl;
	}
#if 1 // only to print buffer organization (if activated add \n before stars)
	for (int p= 0; p < A.col_dist.size(); p++) {
	    typename std::map<int, send_structure >::const_iterator it(A.send_info.find(p));
	    if (it != A.send_info.end())
		out << it->second.indices << "@" << it->second.offset;
	    else
		out << "[]";
	    //out << (it != A.send_indices.end() ? it->second : dense_vector<size_type>());
	}
	out << "\n";
#endif
	if (A.row_dist.rank() < A.row_dist.size()-1) 
	    out << "********";
	out << std::endl;
	start_next(A.row_dist);
	communicator(A.row_dist).barrier();
	return out;
    }

public:
    size_type                      grows, gcols, total_send_size, total_recv_size;
    RowDistribution                row_dist;
    ColDistribution                *cdp, &col_dist;
    
protected:
    local_type                     local_matrix;
public:
    remote_map_type                remote_matrices;
    std::map<int, recv_structure>  recv_info;
    std::map<int, send_structure>  send_info;
};



template <typename DistributedMatrix, 
	  typename Updater = mtl::operations::update_store<typename Collection<DistributedMatrix>::value_type> >
class distributed_inserter
{
    typename DistributedMatrix::row_distribution_type const& row_dist() const { return dist_matrix.row_dist; }
    typename DistributedMatrix::col_distribution_type const& col_dist() const { return dist_matrix.col_dist; }
    int row_rank() const { return row_dist().rank(); }
    int col_rank() const { return col_dist().rank(); }
    int row_size() const { return row_dist().size(); }
    int col_size() const { return col_dist().size(); }

public:
    typedef distributed_inserter                                  self;
    typedef DistributedMatrix                                     dist_matrix_type;
    typedef typename Collection<DistributedMatrix>::size_type     size_type;
    typedef typename Collection<DistributedMatrix>::value_type    value_type;
    typedef typename DistributedCollection<DistributedMatrix>::local_type  local_type;
    typedef typename DistributedCollection<DistributedMatrix>::remote_type remote_type;
    typedef local_type                                            matrix_type; // needed in functors
    typedef inserter<local_type, Updater>                         local_inserter_type;
    typedef inserter<remote_type, Updater>                        remote_inserter_type;
    typedef operations::update_proxy<self, size_type>             proxy_type;
    typedef std::pair< std::pair<size_type, size_type>, value_type > entry_type;
    
    explicit distributed_inserter(DistributedMatrix& dist_matrix, size_type slot_size = 5)
	: dist_matrix(dist_matrix), slot_size(slot_size),
	  local_inserter(dist_matrix.local_matrix, slot_size), 
	  full_remote_matrices(col_size(), (remote_type*) 0),
	  remote_inserters(col_size(), (remote_inserter_type*) 0),
	  send_buffers(row_size()), recv_buffers(row_size())
    {}

    struct col_marker
    {
	typedef typename DistributedMatrix::remote_type remote_type;
	explicit col_marker(const remote_type& A) : A(A), col(A), used_col(num_cols(A), false) {}
	
	template <typename Cursor>
	void operator() (const Cursor& cursor) { used_col[col(*cursor)]= true;	}

	const remote_type&                            A;
	typename mtl::traits::col<remote_type>::type  col;
	std::vector<bool>                             used_col;
    };

    ~distributed_inserter()
    {
	typedef typename dist_matrix_type::recv_structure recv_structure;
	typedef typename dist_matrix_type::send_structure send_structure;

	all_to_all(communicator(col_dist()), send_buffers, recv_buffers);
	for (unsigned p= 0; p < col_size(); p++) {
	    const std::vector<entry_type>& my_buffer= recv_buffers[p];
	    for (unsigned i= 0; i < my_buffer.size(); i++) {
		const entry_type& entry= my_buffer[i];
		update(entry.first.first, entry.first.second, entry.second);
	    }}
	// Finalize insertion
	std::vector< dense_vector<size_type> > index_comp(col_size()), send_indices(col_size()); // compression of column indices	
	dist_matrix.total_recv_size= 0;
	// size_type& ts(dist_matrix.total_recv_size= 0);
	for (unsigned p= 0; p < col_size(); p++)
	    if (remote_inserters[p]) {
		delete remote_inserters[p];
		typename DistributedMatrix::remote_type& A= *full_remote_matrices[p];
		col_marker marker(A);
		for_each_nonzero(A, marker);
		size_type ncols= count(marker.used_col.begin(), marker.used_col.end(), true);
		index_comp[p].change_dim(ncols);
		for (size_type i= 0, pos= 0; i < marker.used_col.size(); i++)
		    if (marker.used_col[i])
			index_comp[p][pos++]= i;
		dist_matrix.recv_info.insert(std::make_pair(int(p), recv_structure(ncols, dist_matrix.total_recv_size)));
		dist_matrix.total_recv_size+= ncols;
		typename traits::reorder<>::type R(reorder(index_comp[p], num_cols(A)));
		dist_matrix.remote_matrices.insert(std::make_pair(int(p), A * trans(R)));
		delete full_remote_matrices[p];
	    }

	dist_matrix.total_send_size= 0;
	all_to_all(communicator(col_dist()), index_comp, send_indices);
	for (unsigned p= 0; p < col_size(); p++)
	    if (size(send_indices[p]) > 0) {
		dist_matrix.send_info.insert(std::make_pair(p, send_structure(send_indices[p], dist_matrix.total_send_size)));
		dist_matrix.total_send_size+= send_indices[p].size();
	    }
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
    DistributedMatrix&                     dist_matrix;
    size_type                              slot_size;
    local_inserter_type                    local_inserter;
    std::vector<remote_type*>              full_remote_matrices; // Remote matrices before compressing columns
    std::vector<remote_inserter_type*>     remote_inserters;
    std::vector<std::vector<entry_type> >  send_buffers, recv_buffers;
};

template <typename DistributedMatrix, typename Updater>
template <typename Modifier>
inline void distributed_inserter<DistributedMatrix, Updater>::modify(size_type row, size_type col, value_type value)
{
    typename DistributedMatrix::row_distribution_type const& row_dist= dist_matrix.row_dist;
    typename DistributedMatrix::col_distribution_type const& col_dist= dist_matrix.col_dist;

    // Somewhere on my proc
    if (row_dist.is_local(row)) {
	size_type local_row= row_dist.global_to_local(row);
	// In local dist or on boundary
	if (col_dist.is_local(col)) {
	    size_type local_col= col_dist.global_to_local(col);
	    local_inserter.modify<Modifier>(local_row, local_col, value);
	} else {
	    int proc= col_dist.on_rank(col);
	    if (!remote_inserters[proc]) { // first insertion for this processor
		// Create matrix and then inserter
		typedef typename DistributedMatrix::remote_type remote_type;
		full_remote_matrices[proc]= new remote_type(row_dist.num_local(num_rows(dist_matrix)),
							    col_dist.num_local(num_cols(dist_matrix), proc));
		remote_inserters[proc]= new remote_inserter_type(*full_remote_matrices[proc], slot_size);
	    }
	    size_type local_col= col_dist.global_to_local(col, proc);
	    remote_inserters[proc]->modify<Modifier>(local_row, local_col, value);
	}
    } else
	send_buffers[row_dist.on_rank(row)].push_back(std::make_pair(std::make_pair(row, col), value));
}

}} // namespace mtl::matrix

#endif // MTL_HAS_MPI

#endif // MTL_MATRIX_DISTRIBUTED_INCLUDE
