// $COPYRIGHT$

#ifndef MTL_MTL_FWD_INCLUDE
#define MTL_MTL_FWD_INCLUDE

/// Main name space for %Matrix Template Library
namespace mtl {

    /// Namespace for tags used for concept-free dispatching
    namespace tag {
	struct row_major;
	struct col_major;

	struct scalar;
	struct vector;
	struct matrix;
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
	template <typename Matrix> struct mat_expr;
	template <typename Matrix> struct dmat_expr;
	template <typename Matrix> struct smat_expr;
	template <typename M1, typename M2, typename SFunctor> struct mat_mat_op_expr;
	template <typename M1, typename M2> struct mat_mat_plus_expr;
	template <typename M1, typename M2> struct mat_mat_minus_expr;

	template <typename Matrix> struct mat_expr;
	template <typename Functor, typename Matrix> struct map_view;
	template <typename Scaling, typename Matrix> struct scaled_view;
	template <typename Matrix>  struct conj_view;
	template <typename Matrix>  struct hermitian_view;
    }

    /// Namespace for vectors and views and %operations exclusively on vectors
    namespace vector {
	template <typename Value, typename Parameters> struct dense_vector;
	template <typename Functor, typename Vector> struct map_view;
	template <typename Vector>  struct conj_view;
	template <typename Scaling, typename Vector> struct scaled_view;
	template <class E1, class E2, typename SFunctor> struct vec_vec_op_expr;
	template <class E1, class E2> struct vec_vec_plus_expr;
	template <class E1, class E2> struct vec_vec_minus_expr;
	template <class E1, class E2, typename SFunctor> struct vec_vec_aop_expr;
	template <class E1, class E2, typename SFunctor> struct vec_scal_aop_expr;
	template <class E1, class E2> struct vec_vec_plus_asgn_expr;
	template <class E1, class E2> struct vec_vec_minus_asgn_expr;
	template <class E1, class E2> struct vec_vec_times_asgn_expr;
    }

    using vector::dense_vector;

    namespace vector {
	template <typename Vector> struct vec_expr;
	template <typename E1, typename E2> struct vec_vec_add_expr;
	template <typename E1, typename E2> struct vec_vec_minus_expr;
    }

    /// Namespace for type %traits
    namespace traits {
	template <typename Value> struct category;
	template <typename Value> struct algebraic_category;

	template <typename Collection> struct value;
	template <typename Collection> struct const_value;
	template <typename Collection> struct row;
	template <typename Collection> struct col;
    }

    /// Namespace for functors with application operator and fully typed paramaters
    namespace tfunctor {
	/// Functor for scaling matrices, vectors and ordinary scalars
	template <typename V1, typename V2, typename AlgebraicCategory = tag::scalar> struct scale;
    }

    /// Namespace for functors with static function apply and fully typed paramaters
    namespace sfunctor {
	template <typename Value, typename AlgebraicCategory = tag::scalar> struct conj;
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

    namespace detail {
	template <typename Matrix, typename ValueType, typename SizeType> struct crtp_matrix_assign;
	template <typename Matrix, typename ValueType, typename SizeType> struct const_crtp_matrix_bracket;
	template <typename Matrix, typename ValueType, typename SizeType> struct crtp_matrix_bracket;
	template <typename Matrix, typename ValueType, typename SizeType> struct const_crtp_base_matrix;
	template <typename Matrix, typename ValueType, typename SizeType> struct crtp_base_matrix;
	template <typename Value, bool OnStack, unsigned Size= 0> struct contiguous_memory_block;
	template <typename Matrix, typename Updater> struct trivial_inserter;
    }

    /// Free function defined for all matrix and vector types
    template <typename Collection> void swap(Collection& c1, Collection& c2);

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
