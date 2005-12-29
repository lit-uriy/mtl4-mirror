// $COPYRIGHT$

#ifndef MTL_COMPRESSED2D_INCLUDE
#define MTL_COMPRESSED2D_INCLUDE

#include <algorithm>
#include <vector>
#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/property_map.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/maybe.hpp>
#include <boost/numeric/mtl/complexity.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>


namespace mtl {

using std::size_t;
using std::vector;

// Forward declarations
struct compressed2D_indexer;
template <typename Elt, typename Parameters> class compressed2D;
template <typename Elt, typename Parameters> class compressed2D_inserter;

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

    compressed_updating_el_cursor () {} 
    compressed_updating_el_cursor (pointer_type me) : super(me) {}

    template <typename Parameters>
    compressed_updating_el_cursor(compressed2D<Elt, Parameters> const& ma, size_t r, size_t c)
	: super(ma.elements() + ma.indexer(ma, r, c))
    {}
};


// Indexing for compressed matrices
struct compressed2D_indexer 
{
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
	return maybe<size_t> (index - &ma.indices[0], *index == minor);
    }

public:
    // Returns the offset if found
    // If not found it returns the position where it would be inserted
    template <class Matrix>
    maybe<size_t> operator() (const Matrix& ma, size_t r, size_t c) const
    {
	// convert into c indices
	typename Matrix::index_type my_index;
	size_t my_r= index::change_from(my_index, r);
	size_t my_c= index::change_from(my_index, c);
	return offset(ma, ma.major_(my_r, my_c), ma.minor_(my_r, my_c));
    }


    // For a given offset the minor can be accessed directly, the major dim has to be searched
    template <class Matrix>
    size_t find_major(const Matrix& ma, size_t offset)
    {
	return std::upper_bound(ma.starts.begin(), ma.starts.end(), offset) - ma.starts.begin();
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
	    super::nnz = new_nnz;
	    super::allocate();
	    indices.resize(super::nnz, 0);
	    data.resize(super::nnz, 0); // ! overloads base matrix
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
	// check if starts has right size
	allocate(last_value - first_value); // ???? 
	// check if nnz and indices has right size
	std::copy(first_value, last_value, super::elements()); // for base matrix
	std::copy(first_value, last_value, data.begin());
	std::copy(first_start, first_start + super::dim1() + 1, starts.begin());
	std::copy(first_index, first_index + super::num_elements(), indices.begin());
    }

    // Consistency check urgently needed !!!

    // Insert function urgently needed !!!
    // void insert(size_t row, size_t col, value_type value)


    friend struct compressed2D_indexer;
    friend struct compressed2D_inserter<Elt, Parameters>;

    indexer_type  indexer;
  protected:
    vector<value_type>      data; // ! overloads base matrix
    vector<size_t>          starts;
    vector<size_t>          indices;
    bool                    inserting;
};

// Additional data structure to insert into compressed 2D matrix type
template <typename Elt, typename Parameters>
class compressed2D_inserter
{
    typedef compressed2D<Elt, Parameters>     matrix_type;
    typedef typename matrix_type::size_type   size_type;
    typedef typename matrix_type::value_type  value_type;

    // stretch matrix rows or columns to slot size (or leave it if equal or greater)
    void stretch();

  public:
    compressed2D_inserter(matrix_type& matrix, size_type slot_size = 5)
	: matrix(matrix), elements(matrix.data), starts(matrix.starts), indices(matrix.indices), 
	  slot_size(slot_size), slot_ends(matrix.dim1()) 
    {
	matrix.inserting = true;
	stretch();
    }

    ~compressed2D_inserter()
    {
	// compress();
	matrix.inserting = true;
    }
	

  protected:
    compressed2D<Elt, Parameters>&      matrix;
    vector<value_type>&                 elements;
    vector<size_type>&                  starts;
    vector<size_type>&                  indices;
    size_type                           slot_size;
    vector<size_type>                   slot_ends;
};

template <typename Elt, typename Parameters>
void compressed2D_inserter<Elt, Parameters>::stretch()
{
    vector<size_type>  new_starts(matrix.dim1() + 1);
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
	    std::copy(&elements[starts[i-1]], &elements[starts[i]], &elements[new_starts[i-1]]);
	    std::copy(&indices[starts[i-1]], &indices[starts[i]], &indices[new_starts[i-1]]);
	} else {
	    std::copy_backward(&elements[starts[i-1]], &elements[starts[i]], &elements[slot_ends[i-1]]);
	    std::copy_backward(&indices[starts[i-1]], &indices[starts[i]], &indices[slot_ends[i-1]]);
	}
    std::swap(starts, new_starts);		    
}



} // namespace mtl

#endif // MTL_COMPRESSED2D_INCLUDE
