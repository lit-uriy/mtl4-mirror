// $COPYRIGHT$

#ifndef MTL_MORTON_DENSE_INCLUDE
#define MTL_MORTON_DENSE_INCLUDE

#include <boost/numeric/mtl/detail/base_cursor.hpp>
#include <boost/numeric/mtl/detail/base_matrix.hpp>
#include <boost/numeric/mtl/matrix_parameters.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/property_map.hpp>
#include <boost/numeric/mtl/range_generator.hpp>
#include <boost/numeric/mtl/detail/range_generator.hpp>
#include <boost/numeric/mtl/complexity.hpp>
#include <boost/numeric/mtl/glas_tags.hpp>


namespace mtl {

// Dense Morton matrix type
template <typename Elt, typename Parameters>
class morton_dense : public detail::base_matrix<Elt, Parameters>, 
		     public detail::contiguous_memory_matrix<Elt, Parameters>
{
    typedef detail::base_matrix<Elt, Parameters>                super;
    typedef detail::contiguous_memory_matrix<Elt, Parameters>   super_memory;
    typedef morton_dense                                        self;
  public:	
    typedef typename Parameters::orientation  orientation;
    typedef typename Parameters::index        index_type;
    typedef typename Parameters::dimensions   dim_type;
    typedef Elt                               value_type;
    typedef std::size_t                       size_type;
     // typedef morton_dense_el_cursor<Elt>       el_cursor_type;  
    // typedef morton_dense_indexer              indexer_type;


  protected:
    void set_nnz()
    {
      this->nnz = this->dim.num_rows() * this->dim.num_cols();
    }
    
    size_type memory_need(size_type rows, size_type cols)
    {
	return 3; // change this
    }
    
  public:
    // if compile time matrix size allocate memory
    morton_dense() : super(), super_memory( memory_need( dim_type().num_rows(), dim_type().num_cols() ) ) {}

    // only sets dimensions, only for run-time dimensions
    explicit morton_dense(mtl::non_fixed::dimensions d) 
	: super(d), super_memory( memory_need( d.num_rows(), d.num_cols() ) ) 
    {
	set_nnz();
    }

    // sets dimensions and pointer to external data
    explicit morton_dense(mtl::non_fixed::dimensions d, value_type* a) 
      : super(d), super_memory(a) 
    { 
        set_nnz();
    }

    // same constructor for compile time matrix size
    // sets dimensions and pointer to external data
    explicit morton_dense(value_type* a) : super(), super_memory(a) 
    { 
	BOOST_ASSERT((dim_type::is_static));
    }

};


} // namespace mtl

#endif // MTL_MORTON_DENSE_INCLUDE
