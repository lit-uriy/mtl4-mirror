// $COPYRIGHT$

#ifndef MTL_ASSIGN_MODES_INCLUDE
#define MTL_ASSIGN_MODES_INCLUDE

#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl { namespace modes {

struct mult_assign_t
{
    static const bool init_to_0= true;

    template <typename T>
    static void init(T& v)
    {
	using math::zero;
	v= zero(v);
    }

    template <typename T, typename U>
    static void update(T& x, const U& y)
    {
	x+= y;
    }
};


struct add_mult_assign_t
{
    static const bool init_to_0= false;

    template <typename T>
    static void init(T& v) {}

    template <typename T, typename U>
    static void update(T& x, const U& y)
    {
	x+= y;
    }
};


struct minus_mult_assign_t
{
    static const bool init_to_0= false;

    template <typename T>
    static void init(T& v) {}

    template <typename T, typename U>
    static void update(T& x, const U& y)
    {
	x-= y;
    }
};


}} // namespace mtl::modes

#endif // MTL_ASSIGN_MODES_INCLUDE
