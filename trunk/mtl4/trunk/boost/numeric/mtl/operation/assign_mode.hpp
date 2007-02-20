// $COPYRIGHT$

#ifndef MTL_ASSIGN_MODE_INCLUDE
#define MTL_ASSIGN_MODE_INCLUDE

#include <boost/numeric/linear_algebra/identity.hpp>

namespace mtl { namespace assign {

struct assign_sum
{
    static const bool init_to_zero= true;

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


struct plus_sum
{
    static const bool init_to_zero= false;

    template <typename T>
    static void init(T& v) {}

    template <typename T, typename U>
    static void update(T& x, const U& y)
    {
	x+= y;
    }
};


struct minus_sum
{
    static const bool init_to_zero= false;

    template <typename T>
    static void init(T& v) {}

    template <typename T, typename U>
    static void update(T& x, const U& y)
    {
	x-= y;
    }
};


}} // namespace mtl::assign

#endif // MTL_ASSIGN_MODE_INCLUDE
