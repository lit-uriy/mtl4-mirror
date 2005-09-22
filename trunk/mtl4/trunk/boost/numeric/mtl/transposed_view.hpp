// $COPYRIGHT$

#ifndef MTL_TRANSPOSED_VIEW_INCLUDE
#define MTL_TRANSPOSED_VIEW_INCLUDE

#include <boost/numeric/mtl/base_types.hpp>

namespace mtl {

template <class Matrix> class transposed_view 
{
    typedef transposed_view               self;
public:	
    typedef Matrix                        other;
    typedef typename transposed_orientation<typename Matrix::orientation>::type orientation;
    typedef typename Matrix::index_type                index_type;
    typedef typename Matrix::value_type                value_type;
    typedef typename Matrix::pointer_type              pointer_type;
    typedef typename Matrix::const_pointer_type        const_pointer_type;
    typedef typename Matrix::key_type                  key_type;
    typedef typename Matrix::size_type                 size_type;
    typedef typename Matrix::dim_type::transposed_type dim_type;
    // typedef typename Matrix::el_cursor_type el_cursor_type;
    // typedef std::pair<el_cursor_type, el_cursor_type> el_cursor_pair;

    transposed_view (other& ref) : ref(ref) {}
    
//     el_cursor_pair elements() const 
//     {
//         return ref.elements();
//     }

    value_type operator() (std::size_t r, std::size_t c) const
    { 
        return ref(c, r); 
    }

    std::size_t dim1() const 
    { 
        return ref.dim2(); 
    }
    std::size_t dim2() const 
    { 
        return ref.dim1(); 
    }
    
    std::size_t offset(const value_type* p) const 
    { 
        return ref.offset(p); 
    }

    const_pointer_type elements() const 
    {
        return ref.elements(); 
    }

    dim_type dimensions() const 
    {
        return ref.dimensions().transpose(); 
    }

    other& ref;
};
  

namespace traits {

    template <class Matrix> struct is_mtl_type<transposed_view<Matrix> > 
    {
	static bool const value= is_mtl_type<Matrix>::value; 
    };

    template <class Matrix> struct matrix_category<transposed_view<Matrix> >
    {
	typedef typename matrix_category<Matrix>::type type;
    };

    template <class Matrix> struct row<transposed_view<Matrix> >
    {
	typedef typename col<Matrix>::type type;
    };

    template <class Matrix> struct col<transposed_view<Matrix> >
    {
	typedef typename row<Matrix>::type type;
    };

    template <class Matrix> struct const_value<transposed_view<Matrix> >
    {
	typedef typename const_value<Matrix>::type type;
    };
    
    template <class Matrix> struct value<transposed_view<Matrix> >
    {
	typedef typename value<Matrix>::type type;
    };

} // namespace traits

template <class Matrix> 
inline typename traits::row<transposed_view<Matrix> >::type
row(const transposed_view<Matrix>& ma)
{
    return col(ma.ref);
}

template <class Matrix>
inline typename traits::col<transposed_view<Matrix> >::type
col(const transposed_view<Matrix>& ma)
{
    return row(ma.ref);
}

template <class Matrix>
inline typename traits::const_value<transposed_view<Matrix> >::type
const_value(const transposed_view<Matrix>& ma)
{
    return const_value(ma.ref);
}

template <class Matrix>
inline typename traits::value<transposed_view<Matrix> >::type
value(const transposed_view<Matrix>& ma)
{
    return value(ma.ref);
}


// ================
// Range generators
// ================

namespace traits
{
    template <class Tag, class Matrix>
    struct range_generator<Tag, transposed_view<Matrix> >
    {
	typedef range_generator<Tag, Matrix>     generator;
	typedef typename generator::complexity   complexity;
	typedef typename generator::type         type;
	static int const                         level = generator::level;
	type begin(transposed_view<Matrix> const& m)
	{
	    return generator().begin(m.ref);
	}
	type end(transposed_view<Matrix> const& m)
	{
	    return generator().end(m.ref);
	}
    };

}


} // namespace mtl

#endif // MTL_TRANSPOSED_VIEW_INCLUDE



