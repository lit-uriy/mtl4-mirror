// $COPYRIGHT$

#ifndef MTL_DIMENSION_INCLUDE
#define MTL_DIMENSION_INCLUDE

namespace mtl { namespace vector {

// Compile time version
namespace fixed {

    template <std::size_t Size>
    struct dimension
    {
	typedef std::size_t  size_type;
	
	static size_type const value= Size;

	size_type size() const
	{
	    return value;
	}

	// to check whether it is static
	static bool const is_static= true;
    };
}

namespace non_fixed {

    template <std::size_t Size>
    struct dimension
    {
	typedef std::size_t  size_type;
	
	static size_type const value= 0; // for compatibility

	dimension() : value(0) {}
	dimension(size_type v) : value(v) {}

	size_type size() const
	{
	    return value;
	}

	// to check whether it is static
	static bool const is_static= false;
    protected:
	size_type value;
    };
}

}} // namespace mtl::vector

#endif // MTL_DIMENSION_INCLUDE
