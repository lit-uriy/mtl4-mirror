// $COPYRIGHT$

#ifndef MTL_COMPRESSED2D_INCLUDE
#define MTL_COMPRESSED2D_INCLUDE

namespace mtl {

using std::size_t;



// Dense 2D matrix type
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
    // typedef compressed2D_indexer                     indexer_type;

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

    // Copies range of values into compressed matrix
    // Only for testing now, will certainly be replaced
    template <typename ValueIterator, typename StartIterator, typename IndexIterator>    
    void raw_copy(ValueIterator first_value, ValueIterator last_value, 
		  StartIterator first_start, IndexIterator first_index)
    {
	allocate(last_value - first_value);
	std::copy(first_value, last_value, matrix.elements());
	std::copy(first_start, first_start + matrix.dim1() + 1, matrix.starts);
	std::copy(first_index, first_index + matrix.num_elements(), matrix);
    }

 protected:
    vector<size_t>          starts;
    vector<size_t>          indices;
};


} // namespace mtl

#endif // MTL_COMPRESSED2D_INCLUDE
