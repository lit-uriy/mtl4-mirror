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
#include <boost/mpi/collectives/all_to_all_sparse.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/concept/collection.hpp>
#include <boost/numeric/mtl/par/distribution.hpp>
#include <boost/numeric/mtl/par/migration.hpp>
#include <boost/numeric/mtl/par/migrate_matrix.hpp>
#include <boost/numeric/mtl/matrix/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/matrix/inserter.hpp>
#include <boost/numeric/mtl/matrix/reorder.hpp>
#include <boost/numeric/mtl/vector/dense_vector.hpp>
#include <boost/numeric/mtl/operation/parallel_utilities.hpp>
#include <boost/numeric/mtl/operation/for_each_nonzero.hpp>
#include <boost/numeric/mtl/operation/trans.hpp>
#include <boost/numeric/mtl/utility/is_row_major.hpp>
#include <boost/numeric/mtl/utility/exception.hpp>

namespace mtl { namespace matrix {


/// Distributed matrices
template <typename Matrix, typename RowDistribution, typename ColDistribution>
class distributed
  : public crtp_matrix_assign< distributed<Matrix, RowDistribution, ColDistribution>, 
			       typename Collection<distributed<Matrix, RowDistribution, ColDistribution> >::value_type,
			       typename Collection<distributed<Matrix, RowDistribution, ColDistribution> >::size_type >
{
    typedef crtp_matrix_assign< distributed<Matrix, RowDistribution, ColDistribution>, 
				typename Collection<distributed<Matrix, RowDistribution, ColDistribution> >::value_type,
				typename Collection<distributed<Matrix, RowDistribution, ColDistribution> >::size_type > assign_base;
  public:
    typedef distributed                              self;
    typedef typename Collection<Matrix>::size_type   size_type;
    typedef typename Collection<Matrix>::value_type  value_type;
    typedef typename Collection<Matrix>::const_reference     const_reference;
    typedef typename OrientedCollection<Matrix>::orientation orientation;

    typedef RowDistribution                          row_distribution_type;
    typedef ColDistribution                          col_distribution_type;
    
    typedef Matrix                                   local_type;
    typedef Matrix                                   remote_type; // Needs specialization
    typedef std::map<int, remote_type>               remote_map_type;
    typedef typename remote_map_type::const_iterator remote_map_const_iterator;

#if 1
    typedef typename Matrix::dim_type                dim_type;   // Shouldn't be needed -> refactor transposed_view!!!         
    typedef typename Matrix::index_type              index_type; // Shouldn't be needed -> refactor transposed_view!!!         
    typedef typename Matrix::key_type                key_type;   // Wrong -> key type should be taken from local matrix not the distributed
#endif

    /// Constructor for matrix with global size grows x gcols and default distribution.
    /** RowDistribution and ColDistribution must have same type. **/
    explicit distributed(size_type grows, size_type gcols) 
      : grows(grows), gcols(gcols), row_dist(grows), cdp(&this->row_dist),
	local_matrix(row_dist.num_local(grows), cdp->num_local(gcols))
    {}

    /// Constructor for matrix with global size grows x gcols and with given distribution.
    /** RowDistribution and ColDistribution must have same type. **/
    explicit distributed(size_type grows, size_type gcols, 
			 const RowDistribution& row_dist) 
      : grows(grows), gcols(gcols), row_dist(row_dist), cdp(&this->row_dist),
	local_matrix(row_dist.num_local(grows), cdp->num_local(gcols))
    {}

    /// Constructor for matrix with global size grows x gcols and with different distributions for rows and columns.
    explicit distributed(size_type grows, size_type gcols, 
			 const RowDistribution& row_dist, const ColDistribution& col_dist) 
      : grows(grows), gcols(gcols), row_dist(row_dist), cdp(new ColDistribution(col_dist)), 
	local_matrix(row_dist.num_local(grows), cdp->num_local(gcols))
    {}

    /// Copy from other types (including expressions)
    template <typename MatrixSrc>
    explicit distributed(const MatrixSrc& src)
      : grows(num_rows(src)), gcols(num_cols(src)), row_dist(row_distribution(src)),
	cdp(&row_distribution(src) == &col_distribution(src) ? &row_dist : new ColDistribution(col_distribution(src))), // refer to row_dist or copy from source
	local_matrix(row_dist.num_local(grows), cdp->num_local(gcols))
    {	*this= src;    }

    // In case new row distribution is to small for global number of columns
    // gcols and row_dist must be set before function is called
    // Row and column distribution must be of same type.
    ColDistribution*
    adapt_col_distribution()
    {
	if (row_dist.max_global() >= gcols) // large enough
	    return &row_dist;
	ColDistribution* new_coll_dist= new ColDistribution(row_dist); // copy constructor
	new_coll_dist->stretch(gcols);
	return new_coll_dist;
    }


    /// Migrating copy
    template <typename MatrixSrc>
    explicit distributed(const MatrixSrc& src, const par::block_migration& migration)
      : grows(num_rows(src)), gcols(num_cols(src)), row_dist(migration.new_distribution()),
	cdp(adapt_col_distribution()), local_matrix(row_dist.num_local(grows), cdp->num_local(gcols))
    {
	migrate_matrix(src, *this, migration);
    }

    ~distributed() { clear_cdp(); }

    using assign_base::operator=; // still need 

    /// Change dimension \p grows global rows times \p gcols global columns 
    /** Potentially changes parametrization of distributions **/
    void change_dim(size_type grows, size_type gcols)
    {
	this->grows= grows; this->gcols= gcols;
	row_dist.resize(grows); cdp->resize(gcols);
	local_matrix.change_dim(row_dist.num_local(grows), cdp->num_local(gcols));
    }

    void check_dim(size_type grows, size_type gcols) const
    {
        MTL_DEBUG_THROW_IF(this->grows * this->gcols != 0 && (this->grows != grows || this->gcols != gcols),
                           incompatible_size());
    }

    void clear_cdp() { if (cdp && cdp != &row_dist) delete cdp; }
    void clear_remote_matrices() { remote_matrices.clear(); recv_info.clear(); send_info.clear(); }

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
	clear_cdp();
	col_dist_assign(src, boost::is_same<RowDistribution, ColDistribution>());
	local_matrix= src.local_matrix;
	// copy remote parts and such
	throw "Implementation not finished yet!";

	return *this;
    }


    /// Leading dimension
    size_type dim1() const { return mtl::traits::is_row_major<self>::value ? grows : gcols; }
    /// Non-leading dimension
    size_type dim2() const { return mtl::traits::is_row_major<self>::value ? gcols : grows; }

    /// Number of non-zeros is only an estimation (for sake of performance), complete nonsense on dense matrices
    size_type nnz() const { return size_type(local_matrix.nnz() * 1.3 * grows / num_rows(local_matrix)); }

    // Decompress column index from buffer index to entire sub-vector range; for internal use only
    size_type decompress_column(size_type col, int p) const
    { return p == cdp->rank() ? col : index_comp.find(p)->second[col];  }

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

    /// Communicator (of row distribution)
    friend inline const boost::mpi::communicator& communicator(const self& d) { return communicator(d.row_dist); }
			      
    template <typename DistMatrix, typename Updater> friend class distributed_inserter;

    /// Number of global rows
    friend inline size_type num_rows(const self& A) { return A.grows; }
    /// Number of global columns
    friend inline size_type num_cols(const self& A) { return A.gcols; }
    /// Global size
    friend inline size_type size(const self& A) { return A.rows * A.gcols; }

    /// Reference to local matrix
    friend inline const local_type& local(const self& A) { return A.local_matrix; }
    friend inline local_type& local(self& A) { return A.local_matrix; }

    friend inline const RowDistribution& row_distribution(const self& A) { return A.row_dist; }
    friend inline const ColDistribution& col_distribution(const self& A) { return *A.cdp; }

    friend inline std::ostream& operator<< (std::ostream& out, const self& A) 
    {
	const ColDistribution  &col_dist(*A.cdp);
	wait_for_previous(A.row_dist);
	const local_type& B= A.local_matrix;
	for (unsigned r= 0; r < num_rows(B); r++) {
	    for (int p= 0; p < col_dist.size(); p++) {
		out << '[';
		if (p == col_dist.rank())
		    for (unsigned c= 0; c < num_cols(B); c++)
			out << B[r][c] << (c < num_cols(B) - 1 ? " " : "");
		else {
		    // Pretty inefficient because redone for every row (refactor when necessary)
		    const unsigned nc= col_dist.num_local(num_cols(A), p);
		    remote_map_const_iterator it(A.remote_matrices.find(p));
		    if (it != A.remote_matrices.end()) {
			const remote_type& C0= it->second; 
			remote_type C(C0 * reorder(A.index_comp.find(p)->second, nc));
			for (unsigned c= 0; c < num_cols(C); c++)
			    out << C[r][c] << (c < num_cols(C) - 1 ? " " : "");
		    } else
			for (unsigned c= 0; c < nc; c++)
			    out << '*' << (c < nc - 1 ? " " : "");
		}
		out << ']';
	    }
	    out << std::endl;
	}
#if 0 // only to print buffer organization
	for (int p= 0; p < col_dist.size(); p++) {
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

    template <typename DistMatrix, typename Visitor> friend void traverse_distributed(const DistMatrix& A, Visitor& vis);
    template <typename Functor, typename M> friend struct map_view;

  protected:
//  public:
    size_type                      grows, gcols, total_send_size, total_recv_size;
    RowDistribution                row_dist;
    ColDistribution                *cdp;
    
    local_type                     local_matrix;
// public:
    remote_map_type                remote_matrices;
    std::map<int, recv_structure>  recv_info;
    std::map<int, send_structure>  send_info;
    std::map<int, dense_vector<size_type> >     index_comp; // compression of columns in receive buffer
};



template <typename DistributedMatrix, 
	  typename Updater = mtl::operations::update_store<typename Collection<DistributedMatrix>::value_type> >
class distributed_inserter
{
    typename DistributedMatrix::row_distribution_type const& row_dist() const { return dist_matrix.row_dist; }
    typename DistributedMatrix::col_distribution_type const& col_dist() const { return *dist_matrix.cdp; }
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

	all_to_all_sparse(communicator(col_dist()), send_buffers, recv_buffers);
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
		dist_matrix.index_comp.insert(std::make_pair(int(p), index_comp[p]));
		typename traits::reorder<>::type R(reorder(index_comp[p], num_cols(A)));
		dist_matrix.remote_matrices.insert(std::make_pair(int(p), A * trans(R)));
		delete full_remote_matrices[p];
	    }

	dist_matrix.total_send_size= 0;
	all_to_all_sparse(communicator(col_dist()), index_comp, send_indices);
	for (unsigned p= 0; p < col_size(); p++)
	    if (size(send_indices[p]) > 0) {
		dist_matrix.send_info.insert(std::make_pair(p, send_structure(send_indices[p], dist_matrix.total_send_size)));
		dist_matrix.total_send_size+= send_indices[p].size();
	    }
    }

    operations::update_bracket_proxy<self, size_type> operator[] (size_type row)
    {	return operations::update_bracket_proxy<self, size_type>(*this, row);    }

    proxy_type operator() (size_type row, size_type col) { return proxy_type(*this, row, col); }

    template <typename Modifier>
    void modify(size_type row, size_type col, value_type val);

    void update(size_type row, size_type col, value_type val) {	modify<Updater>(row, col, val); }

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
    {	return *this << element_matrix_t<Matrix, Rows, Cols>(elements.array, elements.rows, elements.cols);    }

    friend inline const DistributedMatrix& reference(const self& I) { return I.dist_matrix; }

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
    typename DistributedMatrix::col_distribution_type const& col_dist= *dist_matrix.cdp;

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
		set_to_zero(*full_remote_matrices[proc]);
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
