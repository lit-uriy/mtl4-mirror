// $COPYRIGHT$

#ifndef MTL_COMPRESSED2D_INCLUDE
#define MTL_COMPRESSED2D_INCLUDE

#include <algorithm>
#include <vector>
#include <map>
#include <boost/tuple/tuple.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>

#include <boost/numeric/mtl/common_includes.hpp>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/operations/update.hpp>
#include <boost/numeric/mtl/operations/shift_blocks.hpp>

namespace mtl {

// using std::size_t; 
// using std::vector;
// using utilities::maybe;  


// Forward declarations
struct compressed2D_indexer;
template <typename Elt, typename Parameters> class compressed2D;
template <typename Elt, typename Parameters, typename Updater> class compressed2D_inserter;

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
	throw_debug_exception(offset == other.offset && major != other.major,
			      "equal offsets imply equal major\n");
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
	throw_debug_exception(matrix.starts[major+1] < offset, "Inconsistent incrementation!\n");
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

    explicit compressed_minor_cursor(compressed2D<Elt, Parameters> const& matrix, size_t r, size_t c)
	: base(matrix, r, c), matrix(matrix)
    {}

    explicit compressed_minor_cursor(compressed2D<Elt, Parameters> const& matrix, size_t offset) 
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

    compressed2D<Elt, Parameters> const& matrix;
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
    maybe<size_t> offset(const Matrix& ma, size_t major, size_t minor) const 
    {
	const size_t *first = &ma.indices[ ma.starts[major] ],
	             *last = &ma.indices[ ma.starts[major+1] ];
	// if empty row (or column) return start of next one
	if (first == last) 
	    return maybe<size_t> (first - &ma.indices[0], false);
	const size_t *index = std::lower_bound(first, last, minor);
	return maybe<size_t> (index - &ma.indices[0], index != last && *index == minor);
    }

  public:
    // Returns major and minor index in C style (starting with 0)
    template <class Matrix>
    size_pair major_minor_c(const Matrix& ma, size_t row, size_t col) const
    {
	// convert into c indices
	typename Matrix::index_type my_index;
	size_t my_row= index::change_from(my_index, row),
	       my_col= index::change_from(my_index, col);
	return make_pair(ma.major_(my_row, my_col), ma.minor_(my_row, my_col));
    }

    // Returns the offset if found
    // If not found it returns the position where it would be inserted
    template <class Matrix>
    maybe<size_t> operator() (const Matrix& ma, size_t row, size_t col) const
    {
	size_t major, minor;
	boost::tie(major, minor) = major_minor_c(ma, row, col);
	return offset(ma, major, minor);
    }

    // Same as above if internal representation is already known
    template <class Matrix>
    maybe<size_t> operator() (const Matrix& ma, size_pair major_minor) const 
    {
	return offset(ma, major_minor.first, major_minor.second);
    }

    // For a given offset the minor can be accessed directly, the major dim has to be searched
    // Returned in internal (c) representation
    template <class Matrix>
    size_t find_major(const Matrix& ma, size_t offset) const
    {
	throw_debug_exception(ma.starts.empty(), "Major vector can't be empty\n");
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
template <typename Elt, typename Parameters>
class compressed2D 
  : public detail::base_matrix<Elt, Parameters>,
    public detail::const_crtp_base_matrix< compressed2D<Elt, Parameters>, Elt, std::size_t >
{
    typedef std::size_t                              size_t;
    typedef detail::base_matrix<Elt, Parameters>     super;
    typedef compressed2D                             self;
  public:
    typedef Parameters                               parameters;
    typedef typename Parameters::orientation         orientation;
    typedef typename Parameters::index               index_type;
    typedef typename Parameters::dimensions          dimensions;
    typedef Elt                                      value_type;
    // typedef const value_type*                        const_pointer_type;
    typedef compressed_key                           key_type;

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
	    data.resize(this->my_nnz, 0); // ! overloads base matrix
	}
    }

    // if compile time matrix size, we can set the start vector
    explicit compressed2D () 
	: super(), inserting(false)
    {
	if (super::dim_type::is_static) starts.resize(super::dim1() + 1);
    }

    // setting dimension and allocate starting vector
    explicit compressed2D (mtl::non_fixed::dimensions d, size_t nnz = 0) 
      : super(d), inserting(false)
    {
	starts.resize(super::dim1() + 1, 0);
	allocate(nnz);
    }

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

    value_type operator() (size_type row, size_type col) const
    {
        throw_debug_exception(inserting, "Reading data during insertion has undefined behavior!\n");
	maybe<size_type> pos = indexer(*this, row, col);
	return pos ? data[pos] : value_type(0);
    }

    value_type value_from_offset(size_type offset) const
    {
	throw_debug_exception(offset >= this->my_nnz, "Offset larger than matrix!\n");
	return data[offset];
    }

    value_type& value_from_offset(size_type offset)
    {
	throw_debug_exception(offset >= this->my_nnz, "Offset larger than matrix!\n");
	return data[offset];
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
	if (matrix.inserting) throw "Two inserters on same matrix";
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

  private:
    maybe<size_type> matrix_offset(size_pair);
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
    for (size_type i = matrix.dim1(); i > 0; i--)
	if (starts[i] <= new_starts[i-1]) {
	    copy(&elements[starts[i-1]], &elements[starts[i]], &elements[new_starts[i-1]]);
	    copy(&indices[starts[i-1]], &indices[starts[i]], &indices[new_starts[i-1]]);
	} else {
	    copy_backward(&elements[starts[i-1]], &elements[starts[i]], &elements[slot_ends[i-1]]);
	    copy_backward(&indices[starts[i-1]], &indices[starts[i]], &indices[slot_ends[i-1]]);
	}
    swap(starts, new_starts);		    
}

template <typename Elt, typename Parameters, typename Updater>
inline maybe<typename compressed2D_inserter<Elt, Parameters, Updater>::size_type> 
compressed2D_inserter<Elt, Parameters, Updater>::matrix_offset(size_pair mm)
{
    size_type major, minor;
    boost::tie(major, minor) = mm;
    
    const size_t *first = &indices[ starts[major] ],
  	         *last =  &indices[ slot_ends[major] ];
    if (first == last) 
	return maybe<size_t> (first - &indices[0], false);
    const size_t *index = std::lower_bound(first, last, minor);
    return maybe<size_t> (index - &indices[0], index != last && *index == minor);  
}

template <typename Elt, typename Parameters, typename Updater>
inline void compressed2D_inserter<Elt, Parameters, Updater>::update(size_type row, size_type col, value_type val)
{
    using std::copy_backward;

    compressed2D_indexer   indexer;
    size_pair              mm = indexer.major_minor_c(matrix, row, col);
    size_type              major, minor;
    boost::tie(major, minor) = mm;

    maybe<size_type>       pos = matrix_offset(mm);
    // Check if already in matrix and update it
    if (pos) 
	Updater() (elements[pos], val); 
    else {
	size_type& my_end = slot_ends[major];
	// Check if place in matrix to insert there
	if (my_end != starts[major+1]) { 
	    copy_backward(&elements[pos], &elements[my_end], &elements[my_end+1]);
	    copy_backward(&indices[pos], &indices[my_end], &indices[my_end+1]);
	    elements[pos] = val; indices[pos] = minor;
	    my_end++;	    
	    matrix.my_nnz++;      // new entry
	} else {
	    typename map_type::iterator it = spare.find(mm);
	    // If not in map insert it, otherwise update the value
	    if (it == spare.end()) {
		spare.insert(std::make_pair(mm, val));
		matrix.my_nnz++;      // new entry
	    } else 
		Updater() (it->second, val);
	}
    }
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
	maybe<size_type>       pos = matrix_offset(mm);
	size_type&             my_end = slot_ends[major];

	copy_backward(&elements[pos], &elements[my_end], &elements[my_end+1]);
	copy_backward(&indices[pos], &indices[my_end], &indices[my_end+1]);
	elements[pos] = it->second; indices[pos] = minor;
	my_end++;
    }
}

template <typename Elt, typename Parameters>
struct matrix_inserter<compressed2D<Elt, Parameters>, mtl::operations::update_store<Elt> >
  : compressed2D_inserter<Elt, Parameters, mtl::operations::update_store<Elt> >
{
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef compressed2D_inserter<Elt, Parameters, mtl::operations::update_store<Elt> > base;

    explicit matrix_inserter(matrix_type& matrix, size_type slot_size = 5) : base(matrix. slot_size) {}
};

template <typename Elt, typename Parameters, typename Updater>
struct matrix_inserter<compressed2D<Elt, Parameters>, Updater>
  : compressed2D_inserter<Elt, Parameters, Updater>
{
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef compressed2D_inserter<Elt, Parameters, Updater > base;

    explicit matrix_inserter(matrix_type& matrix, size_type slot_size = 5) : base(matrix, slot_size) {}
};


// =============
// Property Maps
// =============

namespace traits 
{
    template <class Elt, class Parameters>
    struct row<compressed2D<Elt, Parameters> >
    {
        typedef typename boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
	  , mtl::detail::major_in_key<compressed2D<Elt, Parameters> >
	  , mtl::detail::indexer_minor_ref<compressed2D<Elt, Parameters> >
	>::type type;  
    };

    template <class Elt, class Parameters>
    struct col<compressed2D<Elt, Parameters> >
    {
        typedef typename boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
	  , mtl::detail::indexer_minor_ref<compressed2D<Elt, Parameters> >
	  , mtl::detail::major_in_key<compressed2D<Elt, Parameters> >
	>::type type;  
    };

    template <class Elt, class Parameters>
    struct const_value<compressed2D<Elt, Parameters> >
    {
	typedef mtl::detail::matrix_offset_const_value<compressed2D<Elt, Parameters> > type;
    };

    template <class Elt, class Parameters>
    struct value<compressed2D<Elt, Parameters> >
    {
	typedef mtl::detail::matrix_offset_value<compressed2D<Elt, Parameters> > type;
    };

} // namespace traits


// ================
// Range generators
// ================

namespace traits
{

// ===========
// For cursors
// ===========

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, compressed2D<Elt, Parameters> >
      : detail::all_offsets_range_generator<compressed2D<Elt, Parameters>,
					    compressed_el_cursor<Elt, Parameters>, 
					    complexity_classes::linear_cached>
    {};

    // Cursor over all rows
    // Supported if row major matrix
    template <typename Elt, typename Parameters>
    struct range_generator<glas::tags::row_t, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
 	  , detail::all_rows_range_generator<compressed2D<Elt, Parameters>, complexity_classes::linear_cached>
 	  , range_generator<glas::tags::unsupported_t, compressed2D<Elt, Parameters> >
        >::type {};


    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::row_t, 2> >
    {
	typedef detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::row_t, 2> cursor_type;
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
    struct range_generator<glas::tags::col_t, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, col_major>
 	  , detail::all_cols_range_generator<compressed2D<Elt, Parameters>, complexity_classes::linear_cached>
 	  , range_generator<glas::tags::unsupported_t, compressed2D<Elt, Parameters> >
        >::type {};


    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_t, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::col_t, 2> >
    {
	typedef detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::col_t, 2> cursor_type;
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
    struct range_generator<glas::tags::major_t, compressed2D<Elt, Parameters> >
      : boost::mpl::if_<
	    boost::is_same<typename Parameters::orientation, row_major>
	  , range_generator<glas::tags::row_t, compressed2D<Elt, Parameters> >
	  , range_generator<glas::tags::col_t, compressed2D<Elt, Parameters> >
        >::type {};


// =============
// For iterators
// =============
 

    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_cit, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::row_t, 2> >
    {
	typedef compressed2D<Elt, Parameters>                                         matrix_type;
	typedef typename matrix_type::size_type                                       size_type;
	typedef typename matrix_type::value_type                                      value_type;
	typedef detail::sub_matrix_cursor<matrix_type, glas::tags::row_t, 2>          cursor;
	
	typedef complexity_classes::linear_cached                                     complexity;
	static int const                                                              level = 1;
	typedef const value_type*                                                     type;
	
	type begin(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, c.key, matrix.begin_col());
	    return &matrix.data[offset];
	}
	
	// returned pointer can pass the end and must only be used for comparison
	type end(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, c.key, matrix.end_col());
	    return &matrix.data[offset];
	}	
    };


    template <class Elt, class Parameters>
    struct range_generator<glas::tags::nz_cit, 
			   detail::sub_matrix_cursor<compressed2D<Elt, Parameters>, glas::tags::col_t, 2> >
    {
	typedef compressed2D<Elt, Parameters>                                         matrix_type;
	typedef typename matrix_type::size_type                                       size_type;
	typedef typename matrix_type::value_type                                      value_type;
	typedef detail::sub_matrix_cursor<matrix_type, glas::tags::col_t, 2>          cursor;
	
	typedef complexity_classes::linear_cached                                     complexity;
	static int const                                                              level = 1;
	typedef const value_type*                                                     type;
	
	type begin(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, matrix.begin_row(), c.key);
	    return &matrix.data[offset];
	}
	
	// returned pointer can pass the end and must only be used for comparison
	type end(cursor const& c)
	{
	    const matrix_type& matrix= c.ref;
	    size_type offset= matrix.indexer(matrix, matrix.end_row(), c.key);
	    return &matrix.data[offset];
	}	
    };


    template <class Elt, class Parameters>
    struct is_mtl_type<compressed2D<Elt, Parameters> > 
    {
	static bool const value= true; 
    };


    // define corresponding type without all template parameters
    template <class Elt, class Parameters>
    struct matrix_category<compressed2D<Elt, Parameters> > 
    {
	typedef tag::sparse type;
    };




} // namespace traits



} // namespace mtl

#endif // MTL_COMPRESSED2D_INCLUDE
