// $COPYRIGHT$

#ifndef MTL_BASE_CASE_TEST_INCLUDE
#define MTL_BASE_CASE_TEST_INCLUDE

#include <algorithm>

namespace mtl { namespace recursion {

// Minimum of dimensions is less or equal to the reference value
struct min_dim_test
{
    min_dim_test(std::size_t comp) : comp(comp) {}

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	return std::min(recurator.get_value().num_rows(), 
			recurator.get_value().num_cols()) 
	       <= comp;
    }

private:
    std::size_t  comp;
};


// Minimum of dimensions is less or equal to the reference value
//   and it can't be split into 2 sub-matrices less or equal the ref value
struct undivisible_min_dim_test
{
    undivisible_min_dim_test(std::size_t comp) : comp(comp) {}

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	std::size_t min_dim= std::min(recurator.get_value().num_rows(), 
				      recurator.get_value().num_cols()),
	            max_dim= std::max(recurator.get_value().num_rows(),
				      recurator.get_value().num_cols());

	return min_dim <= comp && 2 * min_dim > max_dim;
    }

private:
    std::size_t  comp;
};


// Maximum of dimensions is less or equal to the reference value
struct max_dim_test
{
    max_dim_test(std::size_t comp) : comp(comp) {}

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	return std::max(recurator.get_value().num_rows(), 
			recurator.get_value().num_cols()) 
	       <= comp;
    }

private:
    std::size_t  comp;
};


// Same with compile-time reference value
template <unsigned long BaseCaseSize>
struct max_dim_test_static
{
    static const unsigned long base_case_size= BaseCaseSize;

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	return std::max(recurator.get_value().num_rows(), 
			recurator.get_value().num_cols()) 
	       <= BaseCaseSize;
    }
};


// Upper bound of dimensions in recurator is less or equal to the reference value
struct bound_test
{
    bound_test(std::size_t comp) : comp(comp) {}

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	return recurator.bound() <= comp;
    }

private:
    std::size_t  comp;
};


// Same with compile-time reference value
template <unsigned long BaseCaseSize>
struct bound_test_static
{
    static const unsigned long base_case_size= BaseCaseSize;

    template <typename Recurator>
    bool operator() (Recurator const& recurator) const
    {
	return recurator.bound() <= base_case_size;
    }
};



}} // namespace mtl::recursion

#endif // MTL_BASE_CASE_TEST_INCLUDE
