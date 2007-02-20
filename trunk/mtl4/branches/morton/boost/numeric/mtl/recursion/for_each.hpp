// $COPYRIGHT$

#ifndef MTL_FOR_EACH_INCLUDE
#define MTL_FOR_EACH_INCLUDE

namespace mtl { namespace recursion {

// Go recursively down to base case and apply function on it
template <typename Matrix, typename Function, typename BaseCaseTest>
void for_each(matrix_recurator<Matrix> const& recurator, Function const& f, BaseCaseTest const& is_base)
{
    if (is_base(recurator)) 
	f(recurator.get_value());
    else {
	if (!recurator.north_west_empty())
	    for_each(recurator.north_west(), f, is_base);
	if (!recurator.south_west_empty())
	    for_each(recurator.south_west(), f, is_base);
	if (!recurator.north_east_empty())
	    for_each(recurator.north_east(), f, is_base);
	if (!recurator.south_east_empty())
	    for_each(recurator.south_east(), f, is_base);
    }
}


// Non-const version
template <typename Matrix, typename Function, typename BaseCaseTest>
void for_each(matrix_recurator<Matrix>& recurator, Function const& f, BaseCaseTest const& is_base)
{
    typedef matrix_recurator<Matrix> recurator_type;

    if (is_base(recurator)) 
	f(recurator.get_value());
    else {
	if (!recurator.north_west_empty()) {
	    recurator_type  tmp(recurator.north_west());
	    for_each(tmp, f, is_base); }
	if (!recurator.south_west_empty()) {
	    recurator_type  tmp(recurator.south_west());
	    for_each(tmp, f, is_base); }
	if (!recurator.north_east_empty()) {
	    recurator_type  tmp(recurator.north_east());
	    for_each(tmp, f, is_base); }
	if (!recurator.south_east_empty()) {
	    recurator_type  tmp(recurator.south_east());
	    for_each(tmp, f, is_base); }
    }
}


}} // namespace mtl::recursion


#endif // MTL_FOR_EACH_INCLUDE
