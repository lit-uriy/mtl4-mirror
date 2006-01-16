// $COPYRIGHT$

#ifndef MTL_DIMENSIONS_INCLUDE
#define MTL_DIMENSIONS_INCLUDE

#include <iostream>
#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>

namespace mtl {

// dimension is a type for declaring matrix dimensions 
// num_rows() and num_cols() return the number or rows and columns
// is_static says whether it is declared at compile time or not

// Compile time version
namespace fixed
{
    template <std::size_t Row, std::size_t Col>
    struct dimensions
    {
	typedef std::size_t  size_type;

	size_type num_rows() const 
	{
	    return Row;
	}
	size_type num_cols() const 
	{
	    return Col;
	}

	// to check whether it is static
	static bool const is_static= true;

	typedef dimensions<Col, Row> transposed_type;
	transposed_type transpose() const 
	{ 
	    return transposed_type(); 
	}
    };

    template <std::size_t R, std::size_t C>
    std::ostream& operator<< (std::ostream& stream, dimensions<R, C>) 
    {
	return stream << R << 'x' << C; 
    }

} // namespace fixed

namespace non_fixed
{
    struct dimensions
    {
	typedef std::size_t  size_type;

	// some simple constructors
	dimensions() : r(0), c(0) {}
	dimensions(size_type r, size_type c) : r(r), c(c) {}
	dimensions(const dimensions& x) : r(x.r), c(x.c) {}

	dimensions& operator=(const dimensions& x) 
	{
	    r= x.r; c= x.c; return *this; 
	}
	size_type num_rows() const 
	{
	    return r;
	}
	size_type num_cols() const {
	    return c;
	}

	typedef dimensions transposed_type;
	transposed_type transpose() 
	{ 
	    return transposed_type(c, r); 
	}

	static bool const is_static= false;
    protected:
	size_type r, c;
    };

    std::ostream& operator<< (std::ostream& stream, dimensions d) 
    {
	return stream << d.num_rows() << 'x' << d.num_cols(); 
    }

} // namespace non_fixed


#if 0
template <std::size_t Row = 0, std::size_t Col = 0>
struct dimensions
  : public boost::mpl::if_c<
	 Row != 0 && Col != 0
       , struct fixed::dimensions
       , struct non_fixed::dimensions
       >::type
{
    dimensions(std::size_t r, std::size_t c, 
	       typename boost::enable_if_c<Row == 0 || Col == 0>::type* = 0)
	: non_fixed::dimensions(r, c) {}
};
#endif


} // namespace mtl

#endif // MTL_DIMENSIONS_INCLUDE
