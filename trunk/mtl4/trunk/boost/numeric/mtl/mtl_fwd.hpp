// $COPYRIGHT$

#ifndef MTL_MTL_FWD_INCLUDE
#define MTL_MTL_FWD_INCLUDE

namespace mtl {

    namespace tag {
	struct row_major;
	struct col_major;
    }
    using tag::row_major;
    using tag::col_major;

    namespace index {
	struct c_index;
	struct f_index;
    }

    namespace fixed {
	template <std::size_t Rows, std::size_t Cols> struct dimensions;
    }

    namespace non_fixed {
	struct dimensions;
    }

    namespace matrix {
	template <typename Orientation, typename Index, typename Dimensions,
		  bool OnStack, bool RValue>
	struct parameters;
    }

    template <typename Value, typename Parameters> struct dense2D;
    template <typename Value, unsigned long Mask, typename Parameters> 
    struct morton_dense;
    
    template <typename Value, typename Parameters> struct compressed2D;
    template <typename Value, typename Parameters, typename Updater> struct compressed2D_inserter;

    template <typename Matrix> struct transposed_orientation;
    template <typename Matrix> struct transposed_view;

    namespace vector {
	template <typename Value, typename Parameters> struct dense_vector;
    }

    using vector::dense_vector;

    namespace vector {
	template <class E1, class E2> struct vec_vec_add_expr;
	template <class E1, class E2> struct vec_vec_minus_expr;
    }

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
