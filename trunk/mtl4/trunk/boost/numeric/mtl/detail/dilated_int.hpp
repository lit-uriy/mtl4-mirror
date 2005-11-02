// $COPYRIGHT$
//
// Written by Jiahu Deng and Peter Gottschling

#ifndef MTL_DILATED_INT_INCLUDE
#define MTL_DILATED_INT_INCLUDE

#include <boost/numeric/mtl/detail/dilation_table.hpp>
#include <iostream>

namespace mtl { namespace dilated {

    template <typename T>
    struct even_bits
    {
	static T const value = T(-1) / T(3);
    };

    template <typename T>
    struct odd_bits
    {
	static T const value = ~even_bits<T>::value;
    };

    // bin_op1 is binary 'and' for Normalized and binary 'or' for Anti-Normalized
    template <typename T, T BitMask, bool Normalized>
    struct bin_op1_f
    {
	T operator() (T x, T y)
	{
	    return x & y;
	}

	// And is mostly used with original mask
	void masking(T& x)
	{
	    x &= BitMask;
	}
   };

    template <typename T, T BitMask>
    struct bin_op1_f<T, BitMask, false>
    {
	T operator() (T x, T y)
	{
	    return x | y;
	}

	// Or is mostly used with complementary mask
	void masking(T& x)
	{
	    static T const anti_mask = ~BitMask;
	    x |= anti_mask;
	}
	
    };

    template <typename T, T BitMask, bool Normalized>
    struct dilated_int
    {
	typedef T                                   value_type;
	typedef dilated_int<T, BitMask, Normalized> self;

	typedef bin_op1_f<T, BitMask, Normalized>   bin_op1;
	typedef bin_op1_f<T, BitMask, !Normalized>  bin_op2;

	static T const                              bit_mask = BitMask;
	static T const                              anti_mask = ~BitMask;

	// protected:
	
	T i;

    public:

	// Default constructor
	dilated_int()
	{
	    i = Normalized ? 0 : anti_mask;
	}
	

	// Only works for odd and even bits !!!!!!!!!!!!!!!!!!!
	explicit dilated_int(T x)
	{
	    static const T to_switch_on = Normalized ? 0 : anti_mask,
	       	           to_move = anti_mask & 1;
	    T d = dilate_lut[ x & 0xff ] + (dilate_lut[ (x >> 8) & 0xff ] << 16);
	    i = d << to_move | to_switch_on;
	}

	void print_dilated()
	{
	    printf("%x", i);
	}

	self& operator++ ()
	{
	    static T const x = Normalized ? bit_mask : T(-1);
	    i -= x;
	    bin_op1().masking(i);
	    return *this;
	}

	self& operator+= (self const& x)
	{
	    bin_op2().masking(i);
	    i+= x.i;
	    bin_op1().masking(i);
	    return *this;
	}

	self operator+ (self const& x)
	{
	    self tmp(*this);
	    return tmp += x;
	}

	self& operator-= (self const& x)
	{
	    i -= x.i;
	    bin_op1().masking(i);
	    // i = bin_op1(i - x.i, Normalized ? bit_mask : anti_mask);
	    return *this;
	}
	
	self operator- (self const& x)
	{
	    self tmp(*this);
	    return tmp -= x;
	}

    };


}} // namespace mtl::dilated

template <typename T, T BitMask, bool Normalized>
inline std::ostream& operator<< (std::ostream& os, mtl::dilated::dilated_int<T, BitMask, Normalized> d)
{
    os.setf(std::ios_base::hex, std::ios_base::basefield);
    return os << d.i;
}

#endif // MTL_DILATED_INT_INCLUDE
