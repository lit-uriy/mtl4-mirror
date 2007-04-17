// $COPYRIGHT$

#ifndef MTL_MTL_FWD_INCLUDE
#define MTL_MTL_FWD_INCLUDE

/// Main name space for %Matrix Template Library
namespace mtl {

    /// Namespace for tags used for concept-free dispatching
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

    /// Namespace for compile-time parameters, e.g. %matrix dimensions
    namespace fixed {
	template <std::size_t Rows, std::size_t Cols> struct dimensions;
    }

    /// Namespace for run-time parameters, e.g. %matrix dimensions
    namespace non_fixed {
	struct dimensions;
    }

    /// Namespace for matrices and views and operations exclusively on matrices
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

    namespace matrix {
	template <typename Functor, typename Matrix> class map_view;
	template <typename Scaling, typename Matrix> class scaled_view;
	template <typename Matrix>  class conj_view;
    }

    /// Namespace for vectors and views and %operations exclusively on vectors
    namespace vector {
	template <typename Value, typename Parameters> struct dense_vector;
    }

    using vector::dense_vector;

    namespace vector {
	template <class E1, class E2> struct vec_vec_add_expr;
	template <class E1, class E2> struct vec_vec_minus_expr;
    }

    /// Namespace for type %traits
    namespace traits {
	template <typename Matrix> struct category;
	template <typename Matrix> struct value;
	template <typename Matrix> struct const_value;
	template <typename Matrix> struct row;
	template <typename Matrix> struct col;
    }

    /// Namespace for functors with application operator and fully typed paramaters
    namespace tfunctor {
	template <typename V1, typename V2> struct scale;
    }

    /// Namespace for functors with static function apply and fully typed paramaters
    namespace sfunctor {
	template <typename Value> struct conj;
    }

    // Namespace documentations

    /// Namespace for static assignment functors
    namespace assign {}

    /// Namespace for complexity classes
    namespace complexity_classes {}

    /// Namespace for %operations (if not defined in mtl)
    namespace operations {}

    /// Namespace for recursive operations and types with recursive memory layout
    namespace recursion {}

    namespace tag {

	/// Namespace for constant iterator tags
	namespace const_iter {}

	/// Namespace for iterator tags
	namespace iter {}
    }

    /// Namespace for %utilities
    namespace utility {}

    /// Namespace for implementations using recurators
    namespace wrec {}

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
