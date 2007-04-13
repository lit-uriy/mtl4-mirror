// $COPYRIGHT$

#ifndef MTL_MAP_VIEW_INCLUDE
#define MTL_MAP_VIEW_INCLUDE

#include <boost/shared_ptr.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/detail/crtp_base_matrix.hpp>
#include <boost/numeric/mtl/operation/sub_matrix.hpp>


namespace mtl { namespace matrix {

template <typename Functor, typename Matrix> 
class map_view 
  : public detail::crtp_base_matrix< map_view<Functor, Matrix>, 
				     typename Functor::result_type, typename Matrix::size_type >
{
    typedef map_view               self;
public:	
    typedef Matrix                        other;
    typedef typename Matrix::orientation               orientation;
    typedef typename Matrix::index_type                index_type;

    typedef typename Functor::result_type             value_type;
    typedef typename Functor::result_type             const_access_type;
    typedef typename Functor::result_type             const_reference_type;
    typedef typename Matrix::key_type                  key_type;
    typedef typename Matrix::size_type                 size_type;
    typedef typename Matrix::dim_type::transposed_type dim_type;

    map_view (const Functor& functor, other& ref) 
	: functor(functor), ref(ref) 
    {}
    
    map_view (const Functor& functor, boost::shared_ptr<Matrix> p) 
	: functor(functor), my_copy(p), ref(*p)
    {}
    
    const_access_type operator() (size_type r, size_type c) const
    { 
        return Functor::apply(ref(r, c));
    }

    size_type dim1() const 
    { 
        return ref.dim1(); 
    }
    size_type dim2() const 
    { 
        return ref.dim2(); 
    }
    
    dim_type dimensions() const 
    {
        return ref.dimensions();
    }

    size_type begin_row() const
    {
	return ref.begin_row();
    }

    size_type end_row() const
    {
	return ref.end_row();
    }

    size_type num_rows() const
    {
	return ref.num_rows();
    }

    size_type begin_col() const
    {
	return ref.begin_col();
    }

    size_type end_col() const
    {
	return ref.end_col();
    }

    size_type num_cols() const
    {
	return ref.num_cols();
    }
    
    template <typename, typename> friend struct traits::detail::map_value;

protected:
    boost::shared_ptr<Matrix>           my_copy;
public:
    Functor           functor;
    other&            ref;
};
   
}} // namespace mtl::matrix


namespace mtl { namespace traits {

    template <typename Functor, typename Matrix> 
    struct category<matrix::map_view<Functor, Matrix> > 
	: public category<Functor, Matrix>
    {};

    template <typename Functor, typename Matrix> 
    struct row<matrix::map_view<Functor, Matrix> >
	: public row<Matrix>
    {};

    template <typename Functor, typename Matrix> 
    struct col<matrix::map_view<Functor, Matrix> >
	: public col<Matrix>
    {};


    namespace detail {

	template <typename Functor, typename Matrix> 
	struct map_value
	{
	    typedef typename Matrix::key_type                      key_type;
	    typedef typename matrix::map_view<Functor, Matrix>::value_type value_type;
    	
	    map_value(matrix::map_view<Functor, Matrix> const& map_matrix) 
		: its_value(map_matrix.ref) 
	    {}

	    value_type_type operator() (key_type const& key) const
	    {
		return mapped_matrix.functor(its_value(key));
	    }

	  protected:
	    typename const_value<Matrix>::type  its_value;
        };

    } // detail


    template <typename Functor, typename Matrix> 
    struct const_value<matrix::map_view<Functor, Matrix> >
    {
	typedef detail::map_value<Functor, Matrix>  type;
    };


    // ================
    // Range generators
    // ================

    // Use range_generator of original matrix
    template <typename Tag, typename Functor, typename Matrix> 
    struct range_generator<Tag, matrix::map_view<Functor, Matrix> >
	: public range_generator<Tag, Matrix>
    {}


}} // mtl::traits

namespace mtl {

// ==========
// Sub matrix
// ==========

template <typename Matrix>
struct sub_matrix_t< matrix::map_view<Functor, Matrix> >
{
    typedef matrix::map_view<Functor, Matrix>                                     matrix_type;

    // Mapping of sub-matrix type
    typedef typename sub_matrix_t<Matrix>::const_sub_matrix_type                  ref_sub_type;
    typedef matrix::map_view<Functor, ref_sub_type>                               const_sub_matrix_type;

    const_sub_matrix_type operator()(matrix_type const& matrix, size_type begin_r, size_type end_r, 
				     size_type begin_c, size_type end_c)
    {
	typedef boost::shared_ptr<ref_sub_type>                        pointer_type;

	// Submatrix of referred matrix
	// Create a submatrix, whos address will be kept by map_view
	// Functor is copied from view
	pointer_type p(new ref_sub_type(sub_matrix(matrix.ref, begin_r, end_r, begin_c, end_c)));
	return const_sub_matrix_type(matrix.functor, p); 
    }
};




} // namespace mtl


#endif // MTL_MAP_VIEW_INCLUDE
