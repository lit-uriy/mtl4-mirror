// $COPYRIGHT$

#ifndef MTL_COMPRESSED2D_INCLUDE
#define MTL_COMPRESSED2D_INCLUDE

#include <algorithm>
#include <vector>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/utility/common_include.hpp>
#include <boost/numeric/mtl/utility/maybe.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/operation/update.hpp>
#include <boost/numeric/mtl/operation/shift_block.hpp>
#include <boost/numeric/mtl/matrix/mat_expr.hpp>
#include <boost/numeric/mtl/matrix/element_matrix.hpp> 
#include <boost/numeric/mtl/matrix/element_array.hpp> 
#include <boost/numeric/mtl/operation/compute_factors.hpp>

namespace mtl {

// using std::size_t; 
// using std::vector;
// using utilities::maybe;  


// Forward declarations
// template <typename Elt, typename Parameters> class compressed2D;
// template <typename Elt, typename Parameters, typename Updater> class compressed2D_inserter;

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline num_rows(const compressed2D<Value, Parameters>& matrix);

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline num_cols(const compressed2D<Value, Parameters>& matrix);

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline size(const compressed2D<Value, Parameters>& matrix);



struct compressed_key
{
    typedef std::size_t                               size_t;
    typedef compressed_key                            self;
    
    template <typename Elt, typename Parameters>
    explicit compressed_key(compressed2D<Elt, Parameters> const& matrix, size_t offset) : offset(offset)
    {
	std::size_t my_major= matrix.indexer.find_major(matrix, offset);
	major= my_major;
    }

    template <typename Elt, typename Parameters>
    explicit compressed_key(compressed2D<Elt, Parameters> const& matrix, size_t r, size_t c)
    {
	offset= matrix.indexer(matrix, r, c);
	major= matrix.indexer.major_minor_c(matrix, r, c).first;
    }

    compressed_key(compressed_key const& other) 
    {
	offset= other.offset; major= other.major;
    }

    self& operator= (self const& other)
    {
	offset= other.offset; major= other.major;
	return *this;
    }

    bool operator== (compressed_key const& other)
    {
	//if (offset == other.offset && major != other.major) 
	//    std::cout << offset << " " << other.offset << " " << major << " " << other.major << '\n';
	MTL_DEBUG_THROW_IF(offset == other.offset && major != other.major,
			      logic_error("equal offsets imply equal major"));
	return offset == other.offset;
    }

    bool operator!= (compressed_key const& other)
    {
	return !(*this == other);
    }

    size_t       major;
    size_t       offset;
};


// Cursor over every element
template <typename Elt, typename Parameters>
struct compressed_el_cursor 
    : public compressed_key 
{
    typedef Elt                           value_type;
    typedef compressed_key                base;
    typedef compressed_el_cursor          self;
    typedef std::size_t                   size_t;

    explicit compressed_el_cursor(compressed2D<Elt, Parameters> const& matrix, size_t r, size_t c)
	: base(matrix, r, c), matrix(matrix)
    {}

    explicit compressed_el_cursor(compressed2D<Elt, Parameters> const& matrix, size_t offset) 
	: base(matrix, offset), matrix(matrix)
    {}

    compressed_el_cursor(const compressed_el_cursor<Elt, Parameters>& other) 
	: base(other), matrix(other.matrix) 
    {}

    self& operator= (self const& other)
    {
	base::operator=(other);
	return *this;
    }

    self& operator++ ()
    {
	++offset;
	MTL_DEBUG_THROW_IF(matrix.starts[major+1] < offset, runtime_error("Inconsistent incrementation!"));
	while (major < matrix.starts.size()-1 && matrix.starts[major+1] == offset) 
	    ++major;
	return *this;
    }

    base& operator* ()
    {
	return *this;
    }

    compressed2D<Elt, Parameters> const& matrix;
};


// Cursor over every element
template <typename Elt, typename Parameters>
struct compressed_minor_cursor 
    : public compressed_key 
{
    typedef Elt                           value_type;
    typedef compressed_key                base;
    typedef compressed_minor_cursor       self;
    typedef std::size_t                   size_t;

    explicit compressed_minor_cursor(mtl::compressed2D<Elt, Parameters> const& matrix, size_t r, size_t c)
	: base(matrix, r, c), matrix(matrix)
    {}

    explicit compressed_minor_cursor(mtl::compressed2D<Elt, Parameters> const& matrix, size_t offset) 
	: base(matrix, offset), matrix(matrix)
    {}

    compressed_minor_cursor(self const& other) : base(other), matrix(other.matrix) {}

    self& operator= (self const& other)
    {
	base::operator=(other);
	return *this;
    }

    self& operator++ ()
    {
	++offset;
	return *this;
    }

    base& operator* ()
    {
	return *this;
    }

    mtl::compressed2D<Elt, Parameters> const& matrix;
};




// Indexing for compressed matrices
struct compressed2D_indexer 
{
    typedef std::size_t                               size_t;
    typedef size_t                            size_type;
    typedef std::pair<size_type, size_type>   size_pair;
  private:
    // helpers for public functions
    template <class Matrix>
    utilities::maybe<size_t> offset(const Matrix& ma, size_t major, size_t minor) const 
    {
		typedef utilities::maybe<size_t>      result_type;
	assert(ma.starts[major] <= ma.starts[major+1]); // Check sortedness
	assert(ma.starts[major+1] <= ma.my_nnz);        // Check bounds of indices
	// Now we are save to use past-end addresses as iterators

	// Empty matrices are special cases
	if (ma.indices.empty())
		return result_type(0, false);

	const size_t *first = &ma.indices[0] + ma.starts[major],
	             *last = &ma.indices[0] + ma.starts[major+1];
	// if empty row (or column) return start of next one
	if (first == last) 
	    return result_type(first - &ma.indices[0], false);
	const size_t *index = std::lower_bound(first, last, minor);
	return result_type(index - &ma.indices[0], index != last && *index == minor);
    }

  public:
    // Returns major and minor index in C style (starting with 0)
    template <class Matrix>
    size_pair major_minor_c(const Matrix& ma, size_t row, size_t col) const
    {
	using std::make_pair;
	// convert into c indices
	typename Matrix::index_type my_index;
	size_t my_row= index::change_from(my_index, row),
	       my_col= index::change_from(my_index, col);
	return make_pair(ma.major_(my_row, my_col), ma.minor_(my_row, my_col));
    }

    // Returns the offset if found
    // If not found it returns the position where it would be inserted
    template <class Matrix>
    utilities::maybe<size_t> operator() (const Matrix& ma, size_t row, size_t col) const
    {
	size_t major, minor;
	boost::tie(major, minor) = major_minor_c(ma, row, col);
	return offset(ma, major, minor);
    }

    // Same as above if internal representation is already known
    template <class Matrix>
    utilities::maybe<size_t> operator() (const Matrix& ma, size_pair major_minor) const 
    {
	return offset(ma, major_minor.first, major_minor.second);
    }

    // For a given offset the minor can be accessed directly, the major dim has to be searched
    // Returned in internal (c) representation
    template <class Matrix>
    size_t find_major(const Matrix& ma, size_t offset) const
    {
	MTL_DEBUG_THROW_IF(ma.starts.empty(), logic_error("Major vector can't be empty"));
	size_t my_major= std::upper_bound(ma.starts.begin(), ma.starts.end(), offset) - ma.starts.begin();
	return --my_major;
    }

    template <class Matrix>
    size_t minor_from_offset(const Matrix& ma, size_t offset) const
    {
	typedef typename Matrix::index_type my_index;
	return index::change_to(my_index(), ma.indices[offset]);
    }

}; // compressed2D_indexer


// Compressed 2D matrix type
// For now no external data
template <typename Elt, typename Parameters = matrix::parameters<> >
class compressed2D 
  : public detail::base_matrix<Elt, Parameters>,
    public detail::const_crtp_base_matrix< compressed2D<Elt, Parameters>, Elt, std::size_t >,
    public detail::crtp_matrix_assign< compressed2D<Elt, Parameters>, Elt, std::size_t >,
    public matrix::mat_expr< compressed2D<Elt, Parameters> >
{
    typedef std::size_t                              size_t;
    typedef detail::base_matrix<Elt, Parameters>     super;
    typedef compressed2D                             self;
    typedef matrix::mat_expr< compressed2D<Elt, Parameters> >          expr_base;
    typedef detail::crtp_matrix_assign< self, Elt, std::size_t >       assign_base;

  public:
    typedef Parameters                               parameters;
    typedef typename Parameters::orientation         orientation;
    typedef typename Parameters::index               index_type;
    typedef typename Parameters::dimensions          dimensions;
    typedef Elt                                      value_type;
    typedef compressed_key                           key_type;
    // return type of operator() const
    typedef value_type                               const_access_type;

    // typedef const_pointer_type                             key_type;
    typedef size_t                                   size_type;
    typedef compressed2D_indexer                     indexer_type;

    // Only allocation of new data, doesn't copy if already existent
    void allocate(size_t new_nnz)
    {
	if (new_nnz) {
	    this->my_nnz = new_nnz;
	    data.resize(this->my_nnz);
	    indices.resize(this->my_nnz, 0);
	}
    }

    // removes all values; e.g. for set_to_zero
    void make_empty()
    {
	this->my_nnz = 0;
	data.resize(0);
	indices.resize(0);
	std::fill(starts.begin(), starts.end(), 0);
    }

    void change_dim(size_type num_rows, size_type num_cols)
    {
	super::change_dim(mtl::non_fixed::dimensions(num_rows, num_cols));
	starts.resize(this->dim1()+1);
	make_empty();
    }

    // if compile time matrix size, we can set the start vector
    explicit compressed2D () 
	: super(), expr_base(*this), inserting(false)
    {
	if (super::dim_type::is_static) starts.resize(super::dim1() + 1);
    }

    // setting dimension and allocate starting vector
    explicit compressed2D (mtl::non_fixed::dimensions d, size_t nnz = 0) 
      : super(d), expr_base(*this), inserting(false)
    {
	starts.resize(super::dim1() + 1, 0);
	allocate(nnz);
    }

    // setting dimension and allocate starting vector
    compressed2D (size_type num_rows, size_type num_cols, size_t nnz = 0) 
      : super(non_fixed::dimensions(num_rows, num_cols)), expr_base(*this), inserting(false)
    {
	starts.resize(super::dim1() + 1, 0);
	allocate(nnz);
    }

    explicit compressed2D(const self& src)
      : super(non_fixed::dimensions(::mtl::num_rows(src), ::mtl::num_cols(src))), expr_base(*this), inserting(false)
    {
	starts.resize(super::dim1() + 1, 0);
	matrix_copy(src, *this);
    }

    template <typename SrcValue, typename SrcParameters>
    explicit compressed2D(const compressed2D<SrcValue, SrcParameters>& src)
	: super(non_fixed::dimensions(::mtl::num_rows(src), ::mtl::num_cols(src))), expr_base(*this), inserting(false)
    {
	starts.resize(super::dim1() + 1, 0);
	matrix_copy(src, *this);
    }


    // Construction from sum of matrices
    template <typename E1, typename E2>
    explicit compressed2D(const matrix::mat_mat_plus_expr<E1, E2>& src) 
	: expr_base(*this), inserting(false)
    {
	change_dim(num_rows(src.first), num_cols(src.first));
	matrix_copy(src.first, *this);
	*this+= src.second;
    }

    // Construction from difference of matrices
    template <typename E1, typename E2>
    explicit compressed2D(const matrix::mat_mat_minus_expr<E1, E2>& src) 
	: expr_base(*this), inserting(false)
    {
	change_dim(num_rows(src.first), num_cols(src.first));
	matrix_copy(src.first, *this);
	*this-= src.second;
    }

    // Construction from product of matrices
    template <typename E1, typename E2>
    explicit compressed2D(const matrix::mat_mat_times_expr<E1, E2>& src) 
	: expr_base(*this), inserting(false)		
    {
	operation::compute_factors<self, matrix::mat_mat_times_expr<E1, E2> > factors(src);
	change_dim(num_rows(factors.first), num_cols(factors.second));
	mult(factors.first, factors.second, *this);
    }


    // Alleged ambiguity in MSVC 8.0, I need to turn off the warning 
	// Removing the operator ends in run-time error
    self& operator=(const self& src)
    {
	// no self-copy
	if (this == &src) return *this;

	matrix_copy(src, *this);
	return *this;
    }

    using assign_base::operator=;


    // Copies range of values and their coordinates into compressed matrix
    // For brute force initialization, should be used with uttermost care
    // Won't be suitable for distributed matrices, take care of this to this later
    template <typename ValueIterator, typename StartIterator, typename IndexIterator>    
    void raw_copy(ValueIterator first_value, ValueIterator last_value, 
		  StartIterator first_start, IndexIterator first_index)
    {
	using std::copy;

	// check if starts has right size
	allocate(last_value - first_value); // ???? 
	// check if nnz and indices has right size

	copy(first_value, last_value, data.begin());
	copy(first_start, first_start + this->dim1() + 1, starts.begin());
	copy(first_index, first_index + this->nnz(), indices.begin());
    }

    // Consistency check urgently needed !!!

    const_access_type operator() (size_type row, size_type col) const
    {
        MTL_DEBUG_THROW_IF(inserting, logic_error("Reading data during insertion has undefined behavior"));
	utilities::maybe<size_type> pos = indexer(*this, row, col);
	return pos ? data[pos] : value_type(0);
    }

    value_type value_from_offset(size_type offset) const
    {
	MTL_DEBUG_THROW_IF(offset >= this->my_nnz, index_out_of_range("Offset larger than matrix"));
	return data[offset];
    }

    value_type& value_from_offset(size_type offset)
    {
	MTL_DEBUG_THROW_IF(offset >= this->my_nnz, index_out_of_range("Offset larger than matrix"));
	return data[offset];
    }

    friend void swap(self& matrix1, self& matrix2)
    {
	using std::swap;
	static_cast<super&>(matrix1).swap(matrix2);

	swap(matrix1.data, matrix2.data);
	swap(matrix1.starts, matrix2.starts);
	swap(matrix1.indices, matrix2.indices);
	swap(matrix1.inserting, matrix2.inserting);
    }

    friend struct compressed2D_indexer;
    template <typename, typename, typename> friend struct compressed2D_inserter;
    template <typename, typename> friend struct compressed_el_cursor;
    template <typename, typename> friend struct compressed_minor_cursor;

    indexer_type            indexer;
    std::vector<value_type> data; 
  protected:
    std::vector<size_t>     starts;
    std::vector<size_t>     indices;
    bool                    inserting;
};



// ========
// Inserter
// ========

// Additional data structure to insert into compressed 2D matrix type
template <typename Elt, typename Parameters, typename Updater = mtl::operations::update_store<Elt> >
struct compressed2D_inserter
{
    typedef compressed2D_inserter             self;
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef typename matrix_type::value_type  value_type;
    typedef std::pair<size_type, size_type>   size_pair;
    typedef std::map<size_pair, value_type>   map_type;
    typedef operations::update_proxy<self, size_type>   proxy_type;

  private: 
    // stretch matrix rows or columns to slot size (or leave it if equal or greater)
    void stretch();

  public:
    explicit compressed2D_inserter(matrix_type& matrix, size_type slot_size = 5)
	: matrix(matrix), elements(matrix.data), starts(matrix.starts), indices(matrix.indices), 
	  slot_size(slot_size), slot_ends(matrix.dim1()) 
    {
	MTL_THROW_IF(matrix.inserting, runtime_error("Two inserters on same matrix"));
	matrix.inserting = true;
	stretch();
    }

    ~compressed2D_inserter()
    {
	final_place();
	insert_spare();
	matrix.inserting = false;
    }
	
    proxy_type operator() (size_type row, size_type col)
    {
	return proxy_type(*this, row, col);
    }

    void update(size_type row, size_type col, value_type val);

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
	utilities::maybe<typename self::size_type> matrix_offset(size_pair);
    void final_place();
    void insert_spare();

  protected:
    compressed2D<Elt, Parameters>&      matrix;
    std::vector<value_type>&            elements;
    std::vector<size_type>&             starts;
    std::vector<size_type>&             indices;
    size_type                           slot_size;
    std::vector<size_type>              slot_ends;
    map_type                            spare;
};

template <typename Elt, typename Parameters, typename Updater>
void compressed2D_inserter<Elt, Parameters, Updater>::stretch()
{
    using std::copy;
    using std::copy_backward;
    using std::swap;

    std::vector<size_type>  new_starts(matrix.dim1() + 1);
    new_starts[0] = 0;
    for (size_type i = 0; i < matrix.dim1(); i++) {
	size_type entries = starts[i+1] - starts[i];
	slot_ends[i] = new_starts[i] + entries; 
	new_starts[i+1] = new_starts[i] + std::max(entries, slot_size);
    }

    size_type new_total = new_starts[matrix.dim1()];
    elements.resize(new_total);
    indices.resize(new_total);
   
    // copy normally if not overlapping and backward if overlapping
    // i goes down to 1 (not to 0) because i >= 0 never stops for unsigned ;-)
	// &v[i] is replaced by &v[0]+i to enable past-end addresses for STL copy
    for (size_type i = matrix.dim1(); i > 0; i--)
	if (starts[i] <= new_starts[i-1]) {
	    copy(&elements[0] + starts[i-1], &elements[0] + starts[i], &elements[0] + new_starts[i-1]);
	    copy(&indices[0] + starts[i-1], &indices[0] + starts[i], &indices[0] + new_starts[i-1]);
	} else {
		copy_backward(&elements[0] + starts[i-1], &elements[0] + starts[i], &elements[0] + slot_ends[i-1]);
		copy_backward(&indices[0] + starts[i-1], &indices[0] + starts[i], &indices[0] + slot_ends[i-1]);
	}
    swap(starts, new_starts);		    
}

template <typename Elt, typename Parameters, typename Updater>
inline utilities::maybe<typename compressed2D_inserter<Elt, Parameters, Updater>::size_type> 
compressed2D_inserter<Elt, Parameters, Updater>::matrix_offset(size_pair mm)
{
    size_type major, minor;
    boost::tie(major, minor) = mm;
    
	// &v[i] isn't liked by all libs -> &v[0]+i circumvents complaints
    const size_t *first = &indices[0] + starts[major],
  	         *last =  &indices[0] + slot_ends[major];
    if (first == last) 
	return utilities::maybe<size_t> (first - &indices[0], false);
    const size_t *index = std::lower_bound(first, last, minor);
    return utilities::maybe<size_t> (index - &indices[0], index != last && *index == minor);  
}

template <typename Elt, typename Parameters, typename Updater>
inline void compressed2D_inserter<Elt, Parameters, Updater>::update(size_type row, size_type col, value_type val)
{
    using std::copy_backward;

    Updater                updater;  
    compressed2D_indexer   indexer;
    size_pair              mm = indexer.major_minor_c(matrix, row, col);
    size_type              major, minor;
    boost::tie(major, minor) = mm;

    utilities::maybe<size_type>       pos = matrix_offset(mm);
    // Check if already in matrix and update it
    if (pos) 
	updater (elements[pos], val); 
    else {
	size_type& my_end = slot_ends[major];
	// Check if place in matrix to insert there
	if (my_end != starts[major+1]) { 
		copy_backward(&elements[0] + pos.value(), &elements[0] + my_end, &elements[0] + (my_end+1));
		copy_backward(&indices[0] + pos.value(), &indices[0] + my_end, &indices[0] + (my_end+1));
	    elements[pos] = updater.init(val); indices[pos] = minor;
	    my_end++;	    
	    matrix.my_nnz++;      // new entry
	} else {
	    typename map_type::iterator it = spare.find(mm);
	    // If not in map insert it, otherwise update the value
	    if (it == spare.end()) {
		spare.insert(std::make_pair(mm, updater.init(val)));
		matrix.my_nnz++;      // new entry
	    } else 
		updater(it->second, val);
	}
    }
    // std::cout << "inserter update: " << matrix.my_nnz << " non-zero elements, new value is " << elements[pos] << "\n";
}  


template <typename Elt, typename Parameters, typename Updater>
void compressed2D_inserter<Elt, Parameters, Updater>::final_place()
{
    using std::swap;

    size_type          dim1 = matrix.dim1();
    std::vector<size_type>  new_starts(dim1 + 1);
    new_starts[0] = 0;

    typename map_type::iterator it = spare.begin();
    for (size_type i = 0; i < dim1; i++) {
	size_type entries = slot_ends[i] - starts[i];
	while (it != spare.end() && it->first.first == i)
	    entries++, it++;
	new_starts[i+1] = new_starts[i] + entries;
    }

    size_type new_total = new_starts[dim1], old_total = starts[dim1];
    if (new_total > old_total) {
	elements.resize(new_total);
	indices.resize(new_total); }
 
    operations::shift_blocks(dim1, starts, new_starts, slot_ends, elements);
    operations::shift_blocks(dim1, starts, new_starts, slot_ends, indices);

    if (new_total < old_total) {
	elements.resize(new_total);
	indices.resize(new_total); }
 
    for (size_type i = 0; i < dim1; i++)
	slot_ends[i] = new_starts[i] + slot_ends[i] - starts[i];

    swap(starts, new_starts);		    
}

template <typename Elt, typename Parameters, typename Updater>
void compressed2D_inserter<Elt, Parameters, Updater>::insert_spare()
{
    using std::copy_backward;

    for (typename map_type::iterator it = spare.begin(); it != spare.end(); ++it) {
	size_pair              mm = it->first;
	size_type              major = mm.first, minor = mm.second;
	utilities::maybe<size_type>       pos = matrix_offset(mm);
	size_type&             my_end = slot_ends[major];

	// &v[i] see above
	copy_backward(&elements[0] + pos.value(), &elements[0] + my_end, &elements[0] + (my_end+1));
	copy_backward(&indices[0] + pos.value(), &indices[0] + my_end, &indices[0] + (my_end+1));
	elements[pos] = it->second; indices[pos] = minor;
	my_end++;
    }
}

// ================
// Free functions
// ================

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline num_rows(const compressed2D<Value, Parameters>& matrix)
{
    return matrix.num_rows();
}

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline num_cols(const compressed2D<Value, Parameters>& matrix)
{
    return matrix.num_cols();
}

template <typename Value, typename Parameters>
typename compressed2D<Value, Parameters>::size_type
inline size(const compressed2D<Value, Parameters>& matrix)
{
    return matrix.num_cols() * matrix.num_rows();
}





// ================
// Range generators
// ================

namespace traits
{
    // VC 8.0 finds ambiguity with mtl::tag::morton_dense (I wonder why, especially here)
    using mtl::compressed2D;

    // ===========
    // For cursors
    // ===========
        
    template <class Elt, class Parameters>
    struct range_generator<glas::tag::nz, compressed2D<Elt, Parameters> >
      : detail::all_offsets_range_generator<compressed2D<Elt, Parameters>,
					    compressed_el_cursor<Elt, Parameters>, 
					    complexity_classes::linear_cached>
    {};

    // Cursor over all rows
    // Supported if row major matrix
    template <typename Elt, typename Parameters>
    struct range_generator<glas::tag::row, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
 	  , detail::all_rows_range_generator<compressed2D<Elt, Parameters>, complexity_classes::linear_cached>
 	  , range_generator<tag::unsupported, compressed2D<Elt, Parameters> >
        >::type {};


    template <class Elt, class Parameters>
    struct range_generator<glas::tag::nz, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::row, 2> >
    {
	typedef detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::row, 2> cursor_type;
	typedef complexity_classes::linear_cached         complexity;
	typedef compressed_minor_cursor<Elt, Parameters>  type;
	static int const                                  level = 1;

	type begin(cursor_type const& cursor) const
	{
	    return type(cursor.ref, cursor.key, cursor.ref.begin_col());
	}
	type end(cursor_type const& cursor) const
	{
	    return type(cursor.ref, cursor.key, cursor.ref.end_col());
	}
    };

    // Cursor over all columns
    // Supported if column major matrix
    template <typename Elt, typename Parameters>
    struct range_generator<glas::tag::col, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, col_major>
 	  , detail::all_cols_range_generator<compressed2D<Elt, Parameters>, complexity_classes::linear_cached>
 	  , range_generator<tag::unsupported, compressed2D<Elt, Parameters> >
        >::type {};


    template <class Elt, class Parameters>
    struct range_generator<glas::tag::nz, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::col, 2> >
    {
	typedef detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::col, 2> cursor_type;
	typedef complexity_classes::linear_cached         complexity;
	typedef compressed_minor_cursor<Elt, Parameters>  type;
	static int const                                  level = 1;

	type begin(cursor_type const& cursor) const
	{
	    return type(cursor.ref, cursor.ref.begin_row(), cursor.key);
	}
	type end(cursor_type const& cursor) const
	{
	    return type(cursor.ref, cursor.ref.end_row(), cursor.key);
	}
    };

    // Cursor over all rows or columns, depending which one is major
    template <typename Elt, typename Parameters>
    struct range_generator<glas::tag::major, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
	  , range_generator<glas::tag::row, compressed2D<Elt, Parameters> >
	  , range_generator<glas::tag::col, compressed2D<Elt, Parameters> >
        >::type {};


// =============
// For iterators
// =============
 

    template <class Elt, class Parameters>
    struct range_generator<tag::const_iter::nz, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::row, 2> >
    {
	typedef compressed2D<Elt, Parameters>                                         matrix_type;
	typedef typename matrix_type::size_type                                       size_type;
	typedef typename matrix_type::value_type                                      value_type;
	typedef detail::sub_matrix_cursor<matrix_type, glas::tag::row, 2>             cursor;
	
	typedef complexity_classes::linear_cached                                     complexity;
	static int const                                                              level = 1;
	typedef const value_type*                                                     type;
	
	type begin(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, c.key, matrix.begin_col());
	    return &matrix.data[0] + offset;
	}
	
	// returned pointer can pass the end and must only be used for comparison
	type end(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, c.key, matrix.end_col());
	    return &matrix.data[0] + offset;
	}	
    };


    template <class Elt, class Parameters>
    struct range_generator<tag::const_iter::nz, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tag::col, 2> >
    {
	typedef compressed2D<Elt, Parameters>                                         matrix_type;
	typedef typename matrix_type::size_type                                       size_type;
	typedef typename matrix_type::value_type                                      value_type;
	typedef detail::sub_matrix_cursor<matrix_type, glas::tag::col, 2>          cursor;
	
	typedef complexity_classes::linear_cached                                     complexity;
	static int const                                                              level = 1;
	typedef const value_type*                                                     type;
	
	type begin(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, matrix.begin_row(), c.key);
	    return &matrix.data[0] + offset;
	}
	
	// returned pointer can pass the end and must only be used for comparison
	type end(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, matrix.end_row(), c.key);
	    return &matrix.data[0] + offset;
	}	
    };


} // namespace traits



} // namespace mtl

#endif // MTL_COMPRESSED2D_INCLUDE
