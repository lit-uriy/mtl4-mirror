// $COPYRIGHT$

#ifndef MTL_MTL_FWD_INCLUDE
#define MTL_MTL_FWD_INCLUDE

namespace mtl {

    template <typename Value, typename Parameters> struct dense2D;
    template <typename Value, unsigned long Mask, typename Parameters> struct morton_dense;
    
    template <typename Value, typename Parameters> struct compressed2D;

    template <typename Matrix> struct transposed_orientation;
    template <typename Matrix> struct transposed_view;

    namespace traits {
	template <typename Matrix> struct category;
	template <typename Matrix> struct value;
	template <typename Matrix> struct const_value;
	template <typename Matrix> struct row;
	template <typename Matrix> struct col;
    }

} // namespace mtl

#endif // MTL_MTL_FWD_INCLUDE









#if 0
 Once matrices are defined in namespace matrix
namespace mtl {

    namespace matrix {
	
	template <typename Value, typename Parameters> struct dense2D;
	template <typename Value, unsigned long Mask, typename Parameters> struct morton_dense;

	template <typename Value, typename Parameters> struct compressed2D;

#if 0
	template <typename Matrix> struct transposed_orientation;
#endif

    } // namespace matrix

    using matrix::dense2D;
    using matrix::morton_dense;
    using matrix::compressed2D;

} // namespace mtl
#endif
