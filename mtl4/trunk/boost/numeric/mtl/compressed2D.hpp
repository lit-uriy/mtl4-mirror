// $COPYRIGHT$

#ifndef MTL_COMPRESSED2D_INCLUDE
#define MTL_COMPRESSED2D_INCLUDE

namespace mtl {

using std::size_t;

// Forward declarations
template <typename Elt, typename Parameters> class compressed2D;
struct compressed2D_indexer;


// Cursor over every element
template <class Elt> 
struct compressed_el_cursor : public detail::base_cursor<const Elt*> 
{
    typedef Elt                           value_type;
    typedef const value_type*             pointer_type; // ?
    typedef detail::base_cursor<const Elt*> super;

    compressed_el_cursor () {} 
    compressed_el_cursor (pointer_type me) : super(me) {}

    template <typename Parameters>
    compressed_el_cursor(compressed2D<Elt, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}
};

// Cursor over every element
template <class Elt> 
struct compressed_updating_el_cursor : public detail::base_cursor<const Elt*> 
{
    typedef Elt                           value_type;
    typedef const value_type*             pointer_type; // ?
    typedef detail::base_cursor<const Elt*> super;

    compressed_el_cursor () {} 
    compressed_el_cursor (pointer_type me, ) : super(me) {}

    template <typename Parameters>
    compressed_el_cursor(compressed2D<Elt, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}
};


// Indexing for compressed matrices
struct compressed2D_indexer 
    // For a given offset the minor 
    template <class Matrix>
    size_type find_major(const Matrix& ma, size_type offset)
    {
	
    }
}; // compressed2D_indexer


// Compressed 2D matrix type
// For now no external data
template <typename Elt, typename Parameters>
class compressed2D : public detail::base_matrix<Elt, Parameters>
{
    typedef detail::base_matrix<Elt, Parameters>     super;
    typedef compressed2D                             self;
  public:	
    typedef typename Parameters::orientation         orientation;
    typedef typename Parameters::index               index_type;
    typedef typename Parameters::dimensions          dimensions;
    typedef Elt                                      value_type;
    typedef const value_type*                        pointer_type;
    // typedef pointer_type                             key_type;
    typedef size_t                                   size_type;
    // typedef compressed_el_cursor<Elt>                el_cursor_type;  
    typedef compressed2D_indexer                     indexer_type;

    // Only allocation of new data, doesn't copy if already existent
    void allocate(size_t new_nnz)
    {
	if (new_nnz) {
	    nnz = new_nnz;
	    super::allocate();
	    indices.resize(nnz);
	}
    }

    // if compile time matrix size, we can set the start vector
    explicit compressed2D () 
	: super() 
    {
	if (dim_type::is_static) starts.resize(dim1() + 1);
    }

    // setting dimension and allocate starting vector
    explicit compressed2D (mtl::non_fixed::dimensions d, size_t nnz = 0) 
      : super(d) 
    {
	starts.resize(dim1() + 1);
	allocate(nnz);
    }

    // Copies range of values and their coordinates into compressed matrix
    // For brute force initialization, should be used with uttermost care
    // Won't be suitable for distributed matrices, take care of this to this later
    template <typename ValueIterator, typename StartIterator, typename IndexIterator>    
    void raw_copy(ValueIterator first_value, ValueIterator last_value, 
		  StartIterator first_start, IndexIterator first_index)
    {
	// check if starts has right size
	allocate(last_value - first_value);
	// check if nnz and indices has right size
	std::copy(first_value, last_value, matrix.elements());
	std::copy(first_start, first_start + matrix.dim1() + 1, matrix.starts);
	std::copy(first_index, first_index + matrix.num_elements(), matrix);
    }

    // Consistency check urgently needed !!!

    // Insert function urgently needed !!!
    // void insert(size_t row, size_t col, value_type value)


    friend compressed2D_indexer;

 protected:
    vector<size_t>          starts;
    vector<size_t>          indices;
};


} // namespace mtl

#endif // MTL_COMPRESSED2D_INCLUDE
