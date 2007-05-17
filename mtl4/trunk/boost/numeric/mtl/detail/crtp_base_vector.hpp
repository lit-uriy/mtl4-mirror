// $COPYRIGHT$

#ifndef MTL_CRTP_BASE_VECTOR_INCLUDE
#define MTL_CRTP_BASE_VECTOR_INCLUDE



namespace mtl { namespace vector {


/// Base class to provide vector assignment operators generically 
template <typename Vector, typename ValueType, typename SizeType>
struct crtp_vector_assign
{

    template <class E>
    vec_vec_asgn_expr<Vector, E> operator=( vec_expr<E> const& e )
    {
	static_cast<Vector*>(this)->check_consistent_shape(e);
	return vec_vec_asgn_expr<Vector, E>( static_cast<Vector&>(*this), e.ref );
    }


}



template <typename Vector, typename ValueType, typename SizeType>
struct crtp_base_vector 
    : public crtp_vector_assign<Vector, ValueType, SizeType>
{};


}} // namespace mtl::vector

#endif // MTL_CRTP_BASE_VECTOR_INCLUDE
