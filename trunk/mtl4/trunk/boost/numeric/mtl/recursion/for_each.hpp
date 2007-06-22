// $COPYRIGHT$

#ifndef MTL_FOR_EACH_INCLUDE
#define MTL_FOR_EACH_INCLUDE

namespace mtl { namespace recursion {

// Go recursively down to base case and apply function on it
template <typename Matrix, typename Function, typename BaseCaseTest>
void for_each(matrix_recursator<Matrix> const& recursator, Function const& f, BaseCaseTest const& is_base)
{
    if (is_base(recursator)) 
	f(recursator.get_value());
    else {
	if (!recursator.north_west_empty())
	    for_each(recursator.north_west(), f, is_base);
	if (!recursator.south_west_empty())
	    for_each(recursator.south_west(), f, is_base);
	if (!recursator.north_east_empty())
	    for_each(recursator.north_east(), f, is_base);
	if (!recursator.south_east_empty())
	    for_each(recursator.south_east(), f, is_base);
    }
}


// Non-const version
template <typename Matrix, typename Function, typename BaseCaseTest>
void for_each(matrix_recursator<Matrix>& recursator, Function const& f, BaseCaseTest const& is_base)
{
    typedef matrix_recursator<Matrix> recursator_type;

    if (is_base(recursator)) 
	f(recursator.get_value());
    else {
	if (!recursator.north_west_empty()) {
	    recursator_type  tmp(recursator.north_west());
	    for_each(tmp, f, is_base); }
	if (!recursator.south_west_empty()) {
	    recursator_type  tmp(recursator.south_west());
	    for_each(tmp, f, is_base); }
	if (!recursator.north_east_empty()) {
	    recursator_type  tmp(recursator.north_east());
	    for_each(tmp, f, is_base); }
	if (!recursator.south_east_empty()) {
	    recursator_type  tmp(recursator.south_east());
	    for_each(tmp, f, is_base); }
    }
}


}} // namespace mtl::recursion


#endif // MTL_FOR_EACH_INCLUDE
