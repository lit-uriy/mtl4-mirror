// $COPYRIGHT$

#ifndef MTL_VECTOR_MAP_VIEW_INCLUDE
#define MTL_VECTOR_MAP_VIEW_INCLUDE

#include <boost/shared_ptr.hpp>
#include <boost/numeric/mtl/utility/category.hpp>
#include <boost/numeric/mtl/utility/range_generator.hpp>
#include <boost/numeric/mtl/utility/property_map.hpp>
#include <boost/numeric/mtl/operation/sfunctor.hpp>
#include <boost/numeric/mtl/operation/tfunctor.hpp>
#include <boost/numeric/mtl/operation/conj.hpp>
#include <boost/numeric/mtl/vector/vec_expr.hpp>



namespace mtl { namespace vector { namespace detail {
    // Forward declaration for friend declaration
    template <typename, typename> struct map_value;
}}}

namespace mtl { namespace vector {

template <typename Functor, typename Vector> 
class map_view 
  : public vec_expr< map_view<Functor, Vector> >
{
    typedef map_view                                   self;
    typedef vector::vec_expr< self >                   expr_base;
public:	
    typedef Vector                                     other;

    typedef typename Functor::result_type              value_type;
    typedef typename Functor::result_type              const_access_type;
    typedef typename Functor::result_type              const_reference_type;
    typedef typename Vector::key_type                  key_type;
    typedef typename Vector::size_type                 size_type;

    // Deprecated: concept map defined in collection.hpp
    typedef typename Vector::orientation               orientation;
    

    map_view (const Functor& functor, const other& ref) 
	: expr_base(*this), functor(functor), ref(ref) 
    {}
    
    map_view (const Functor& functor, boost::shared_ptr<Vector> p) 
	: expr_base(*this), functor(functor), my_copy(p), ref(*p)
    {}

    const_reference_type operator() (size_type i) const
    { 
        return functor(ref(i));
    }

    const_reference_type operator[] (size_type i) const
    { 
        return functor(ref[i]);
    }
    
    void delay_assign() const {}
    
    template <typename, typename> friend struct detail::map_value;

  protected:
    boost::shared_ptr<Vector>           my_copy;
  public:
    Functor           functor;
    const other&      ref;

};

// ================
// Free functions
// ================

template <typename Functor, typename Vector>
typename map_view<Functor, Vector>::size_type
inline size(const map_view<Functor, Vector>& view)
{
    return size(view.ref);
}


    namespace detail {

	template <typename Functor, typename Vector> 
	struct map_value
	{
	    typedef typename Vector::key_type                      key_type;
	    typedef typename vector::map_view<Functor, Vector>::value_type value_type;
    	
	    map_value(vector::map_view<Functor, Vector> const& map_vector) 
		: map_vector(map_vector), its_value(map_vector.ref) 
	    {}

	    value_type operator() (key_type const& key) const
	    {
		return map_vector.functor(its_value(key));
	    }

	  protected:
	    vector::map_view<Functor, Vector> const&   map_vector;
	    typename ::mtl::traits::const_value<Vector>::type its_value;
        };

    } // detail

}} // namespace mtl::vector

    // typedef typename OrientatedCollection<Vector>::orientation orientation;  // concept-style

namespace mtl { 

#if 0
    // orientation in concept-style; not yet implemented
    template <typename Functor, typename Vector> 
    struct OrientatedCollection< vector::map_view<Functor, Vector> >
    {
	typedef typename OrientatedCollection< Vector >::type type;
    };
    // and concept map
#endif
}

namespace mtl { namespace traits {

    // ================
    // Property maps
    // ================

    template <typename Functor, typename Vector> 
    struct index<vector::map_view<Functor, Vector> >
	: public index<Vector>
    {};

    template <typename Functor, typename Vector> 
    struct const_value<vector::map_view<Functor, Vector> >
    {
	typedef vector::detail::map_value<Functor, Vector>  type;
    };


    // ================
    // Range generators
    // ================

    // Use range_generator of original vector
    template <typename Tag, typename Functor, typename Vector> 
    struct range_generator<Tag, vector::map_view<Functor, Vector> >
	: public range_generator<Tag, Vector>
    {};

}} // mtl::traits

namespace mtl { namespace vector {

template <typename Scaling, typename Vector>
struct scaled_view
    : public map_view<tfunctor::scale<Scaling, typename Vector::value_type>, Vector>
{
    typedef tfunctor::scale<Scaling, typename Vector::value_type>  functor_type;
    typedef map_view<functor_type, Vector>                         base;

    scaled_view(const Scaling& scaling, const Vector& vector)
	: base(functor_type(scaling), vector)
    {}
    
    scaled_view(const Scaling& scaling, boost::shared_ptr<Vector> p)
	: base(functor_type(scaling), p)
    {}
};


template <typename Vector>
struct conj_view
    : public map_view<sfunctor::conj<typename Vector::value_type>, Vector>
{
    typedef sfunctor::conj<typename Vector::value_type>            functor_type;
    typedef map_view<functor_type, Vector>                         base;

    conj_view(const Vector& vector)
	: base(functor_type(), vector)
    {}
    
    conj_view(boost::shared_ptr<Vector> p)
	: base(functor_type(), p)
    {}
};

}} // namespace mtl::vector


#endif // MTL_VECTOR_MAP_VIEW_INCLUDE
