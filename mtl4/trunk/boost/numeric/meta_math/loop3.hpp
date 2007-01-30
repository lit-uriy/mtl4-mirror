// $COPYRIGHT$

#ifndef META_MATH_LOOP3_INCLUDE
#define META_MATH_LOOP3_INCLUDE

// See below for example

namespace meta_math {

template <unsigned long Index0, unsigned long Max0, unsigned long Index1, unsigned long Max1,
	  unsigned long Index2, unsigned long Max2>
struct loop3
{
    static unsigned long const index0= Index0 - 1, next_index0= Index0,
	                       index1= Index1 - 1, next_index1= Index1,
            	               index2= Index2 - 1, next_index2= Index2 + 1;
};


template <unsigned long Index0, unsigned long Max0, unsigned long Index1, unsigned long Max1, 
	  unsigned long Max2>
struct loop3<Index0, Max0, Index1, Max1, Max2, Max2>
{
    static unsigned long const index0= Index0 - 1, next_index0= Index0,
	                       index1= Index1 - 1, next_index1= Index1 + 1,
            	               index2= Max2 - 1, next_index2= 1;
};


template <unsigned long Index0, unsigned long Max0, unsigned long Max1, unsigned long Max2>
struct loop3<Index0, Max0, Max1, Max1, Max2, Max2>
{
    static unsigned long const index0= Index0 - 1, next_index0= Index0 + 1,
	                       index1= Max1 - 1, next_index1= 1,
            	               index2= Max2 - 1, next_index2= 1;
};


template <unsigned long Max0, unsigned long Max1, unsigned long Max2>
struct loop3<Max0, Max0, Max1, Max1, Max2, Max2>
{
    static unsigned long const index0= Max0 - 1,
	                       index1= Max1 - 1,
            	               index2= Max2 - 1;
};




#if 0

// ============================
// Use the meta loop like this:
// ============================


template <unsigned long Index0, unsigned long Max0, unsigned long Index1, unsigned long Max1,
	  unsigned long Index2, unsigned long Max2>
struct loop3_trace : public loop3<Index0, Max0, Index1, Max1, Index2, Max2>
{
    typedef loop3<Index0, Max0, Index1, Max1, Index2, Max2> base;
    typedef loop3_trace<base::next_index0, Max0, base::next_index1, Max1, base::next_index2, Max2> next_t;

    void operator() ()
    {
	std::cout << this->index0 << " : " << this->index1 << " : " << this->index2 << "\n";
	next_t() ();
    }  
};


template <unsigned long Max0, unsigned long Max1, unsigned long Max2>
struct loop3_trace<Max0, Max0, Max1, Max1, Max2, Max2>
    : public loop3<Max0, Max0, Max1, Max1, Max2, Max2>
{
    void operator() ()
    {
	std::cout << this->index0 << " : " << this->index1 << " : " << this->index2 << "\n";
    }  
};

#endif

} // namespace meta_math

#endif // META_MATH_LOOP3_INCLUDE
