// $COPYRIGHT$

#ifndef MTL_PRINT_INCLUDE
#define MTL_PRINT_INCLUDE

#include <iostream>
#include <boost/numeric/mtl/mtl_fwd.hpp>
#include <boost/numeric/mtl/operation/print_matrix.hpp>
#include <boost/numeric/mtl/operation/print_vector.hpp>
#include <boost/numeric/mtl/utility/tag.hpp>

namespace mtl {

    namespace detail {

	template <typename Value>
	inline std::ostream&
	print(Value const& value, tag::matrix, std::ostream& out= std::cout, int width= 3, int precision= 2)
	{
	    return print_matrix(value, out, width, precision);
	}

	template <typename Value>
	inline std::ostream&
	print(Value const& value, tag::vector, std::ostream& out= std::cout, int width= 3, int precision= 2)
	{
	    return print_vector(value, out, width, precision);
	}

    } // namespace detail


    template <typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, const matrix::mat_expr<Matrix>& expr)
    {
	return print_matrix(expr.ref, out, 3, 2);
    }


    template <typename Vector>
    inline std::ostream& operator<< (std::ostream& out, const vector::vec_expr<Vector>& expr)
    {
	return print_vector(expr.ref, out, 0, 0);
    }


    template <typename Value>
    inline std::ostream&
    print(Value const& value, std::ostream& out= std::cout, int width= 3, int precision= 2)
    {
	return detail::print(value, typename traits::category<Value>::type(), out, width, precision);
    }



#if 0


    template <typename Value, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, dense2D<Value, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Value, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, compressed2D<Value, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Value, unsigned long Mask, typename Parameter>
    inline std::ostream& operator<< (std::ostream& out, morton_dense<Value, Mask, Parameter> const& value) 
    {
	return print(value, out);
    }

    template <typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, transposed_view<Matrix> const& value) 
    {
	return print(value, out);
    }

    template <typename Functor, typename Matrix>
    inline std::ostream& operator<< (std::ostream& out, matrix::map_view<Functor, Matrix> const& value) 
    {
	return print(value, out);
    }

#endif




// ======================
// use formatting with <<
// ======================


    namespace detail {

	template <typename Collection>
	struct with_format_t
	{
	    explicit with_format_t(const Collection& collection, int width, int precision) 
		: collection(collection), width(width), precision(precision)
	    {}

	    const Collection& collection;
	    int width, precision;
	};

    } // detail


    template <typename Collection>
    inline detail::with_format_t<Collection> with_format(const Collection& collection, int width= 3, int precision= 2)
    {
	return detail::with_format_t<Collection>(collection, width, precision);
    }


    template <typename Collection>
    inline std::ostream& operator<< (std::ostream& out, detail::with_format_t<Collection> const& value) 
    {
	return print(value.collection, out, value.width, value.precision);
    }
    

} // namespace mtl

#endif // MTL_PRINT_INCLUDE
