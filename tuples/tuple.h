#include "boost/type_traits.hpp"
#include <utility> // for pair.hpp


namespace std {

  using ::boost::is_function; 

  namespace tuples {

    // Some helper tools

    // compile time if statement, hopefully this will be part of the
    // standard elsewhere

    template <bool If, class Then, class Else> struct ct_if {
      typedef Then type;
    };

    template <class Then, class Else> struct ct_if<false, Then, Else> {
      typedef Else type;
    };

    // Member variables cannot be of type void or of a function type.
    // (TODO: Find a reference in the standard)
    // Just defining such a class (e.g. by instantiating a template)
    // is an error.

    // These traits templates map void types and plain function types
    // to a type that does not cause the class definition to fail, only
    // creating an object of such class fails.

    // The reationale is to allow one to write tuple types with void or
    // function types as elements, even though it is not possible to 
    // instantiate such object.

    //  E.g: typedef tuple<void> some_type; // ok
    //  but: some_type x; // fails

    template <class T> class non_storeable_type {
      non_storeable_type();
    };

    template <class T> struct non_storeable_type {
      typedef typename ct_if<
        is_function<T>::value, non_storeable_type<T>, T
      >::type type;
    };

    template <> struct non_storeable_type<void> {
      typedef non_storeable_type<void> type; 
    };

    template <class T>
    class wrapper {
      typedef typename add_reference<
        typename add_const<T>::type
      >::type par_t;

      T elem;
    public:
  
      typedef T element_t;

      // take all parameters as const rererences. 
      // Note that non-const references
      // stay as they are.

      identity(par_t t) : elem(t) {}
      T& unwrap() { return elem; }
    };


    // This is add_reference
    template <class T> struct access_traits_non_const {
      typedef T& type;
    };
    template <class T> struct access_traits_non_const<T&> {
      typedef T& type;
    };
    

    template <class T>
    wrapper 
    // This is add_const, add refernce
    template <class T> struct access_traits_const {
      typedef const T& type;
    };
    
    template <class T> struct access_traits_const<T&> {
      typedef T& type;
    };
    
    
    // used as the tuple constructors parameter types
    // Rationale: non-reference tuple element types can be cv-qualified.
    // It should be possible to initialize such types with temporaries,
    // and when binding temporaries to references, the reference must
    // be non-volatile and const. 8.5.3. (5)

    template <class T> struct access_traits_parameter {
      typedef const typename boost::remove_cv<T>::type& type;
    };
    
    template <class T> struct access_traits_parameter<T&> {
      typedef T& type;   
    };


  }
    
  template <class H, class T>
  class cons {
    
    typedef typename 
    tuples::non_storeable_type<head_type>::type stored_head_type;

    typedef typename add_reference<stored_head_type>::type ref_head;
    typedef typename add_reference<tail_type>::type ref_tail;

    typedef typename add_const<stored_head_type>::type const_head;
    typedef typename add_const<stored_tail_type>::type const_tail;

    typedef typename add_reference<const_head>::type const_ref_head;
    typedef typename add_reference<const_tail>::type const_ref_tail;


  public:
    typedef H head_type;
    typedef T tail_type;
    
    stored_head_type head; 
    tail_type tail;
    
    // stored_head_type&
    ref_head get_head() { return head; }
      
    // tail_type&
    ref_tail get_tail() { return tail; }
      
   
    // const stored_head_type&
    const_ref_head  get_head() const { return head; }
      
    // const tail_type&
    const_ref_tail get_tail() const { return tail; }

      
    cons() : head(), tail() {}
      
    cons(typename const remove_volatile<stored_head_type>::type& h,
         const tail_type& t);

    template <class H2, class T2>
    cons( const cons<H2, T2>& u ) : head(u.head), tail(u.tail) {}

    template <class HT2, class TT2>
    cons& operator=( const cons<HT2, TT2>& u ) { 
      head = u.head; 
      tail = u.tail; 
      return *this; 
    }
    
    // must define assignment operator explicitly, implicit version is 
    // illformed if HT is a reference (12.8. (12))
    cons& operator=(const cons& u) { 
      head = u.head; tail = u.tail;  return *this; 
    }
    
    template <class T1, class T2>
    cons& operator=( const std::pair<T1, T2>& u ) { 
      BOOST_STATIC_ASSERT(length<cons>::value == 2); // check length = 2
      head = u.first; tail.head = u.second; return *this;
    }

  protected:
    template <class T1, class T2, class T3, class T4, class T5, 
      class T6, class T7, class T8, class T9, class T10>
    cons( T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, 
	  T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 ) 
      : head (t1), 
      tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, detail::cnull())
    {}

    template <class T2, class T3, class T4, class T5, 
      class T6, class T7, class T8, class T9, class T10>
    cons( const null_type& t1, T2& t2, T3& t3, T4& t4, T5& t5, 
	  T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 ) 
      : head (), 
      tail (t2, t3, t4, t5, t6, t7, t8, t9, t10, detail::cnull())
    {}

  };    
    

  template <class HT>
  struct cons<HT, null_type> {
    
    typedef typename non_storeable_type<head_type>::type stored_head_type;
    
    typedef typename add_reference<stored_head_type>::type ref_head;

    typedef typename add_const<stored_head_type>::type const_head;

    typedef typename add_reference<const_head>::type const_ref_head;

  public:
    
    typedef HT head_type;
    typedef null_type tail_type;

    stored_head_type head;

    // stored_head_type&
    ref_head get_head() { return head; }
    
    null_type get_tail() { return null_type(); }
      
   
    // const stored_head_type&
    const_ref_head  get_head() const { return head; }
    
    null_type get_tail() const { return null_type(); }

    cons() : head() {}

    cons(typename const remove_volatile<stored_head_type>::type& h,
	 const null_type& = null_type())
      : head (h) {}  

    template <class HT2>
    cons( const cons<HT2, null_type>& u ) : head(u.head) {}
  
    template <class HT2>
    cons& operator=(const cons<HT2, null_type>& u ) { 
      head = u.head; 
      return *this; 
    }

    // must define assignment operator explicitely, implicit version 
    // is illformed if HT is a reference
    cons& operator=(const cons& u) { head = u.head; return *this; }

    template<class T1>
    cons(T1& t1, const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&)
      : head (t1) {}

    cons(const null_type& t1, 
	 const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&)
      : head () {}
  };


  namespace tuples {

    // Tuple to cons mapper --------------------------------------------------
    template <class T0, class T1, class T2, class T3, class T4, 
      class T5, class T6, class T7, class T8, class T9>
    struct map_tuple_to_cons
    {
      typedef cons<T0, 
	typename map_tuple_to_cons<T1, T2, T3, T4, T5, 
	                           T6, T7, T8, T9, null_type>::type
      > type;
    };

    // The empty tuple is a null_type
    template <>
    struct map_tuple_to_cons<null_type, null_type, null_type, null_type, 
                             null_type, null_type, null_type, null_type, 
                             null_type, null_type>
    {
      typedef null_type type;
    };

    // tuple default argument wrappers ---------------------------------------
    // Work for non-reference types, intentionally not for references
    template <class T>
    struct def {
      
      // Non-class temporaries cannot have qualifiers.
      // To prevent f to return for example const int, we remove cv-qualifiers
      // from all temporaries.
      static typename boost::remove_cv<T>::type f() { return T(); }
    };
    
    // This is just to produce a more informative error message
    // The code would fail in any case
    template<class T, int N>
    struct def<T[N]> {
      static T* f() {
	return generate_error<T[N]>::arrays_are_not_valid_tuple_elements; }
    };

    template <class T> struct generate_error {};
    
    template <class T>
    struct def<T&> {
      static T& f() {
	return generate_error<T>::no_default_values_for_reference_types;
      }
};

  } // end tuples

  // -------------------------------------------------------------------
  // -- tuple ------------------------------------------------------
  
  // - tuple forward declaration -------------------------------------------
  template <
  class T0 = null_type, class T1 = null_type, class T2 = null_type, 
  class T3 = null_type, class T4 = null_type, class T5 = null_type, 
  class T6 = null_type, class T7 = null_type, class T8 = null_type, 
  class T9 = null_type>
class tuple; 

  template <class T> def() { 
  };


  template <class T0, class T1, class T2, class T3, class T4, 
            class T5, class T6, class T7, class T8, class T9>

  class tuple : 
    public detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type   
  {
  public:
    typedef typename 
    detail::map_tuple_to_cons<T0, T1, T2, T3, T4, T5, T6, T7, T8, T9>::type 
    base;
    typedef typename base::head_type head_type;
    typedef typename base::tail_type tail_type;  


    // access_traits_parameter<T>::type takes non-reference types as const T& 
    tuple() {}
  

    explicit 
    tuple(wrapper<T0> t0 = def<T0>::f(),
	  wrapper<T1> t1 = def<T1>::f(),
	  wrapper<T2> t2 = def<T2>::f(),
	  wrapper<T3> t3 = def<T3>::f(),
	  wrapper<T4> t4 = def<T4>::f(),
	  wrapper<T5> t5 = def<T5>::f(),
	  wrapper<T6> t6 = def<T6>::f(),
	  wrapper<T7> t7 = def<T7>::f(),
	  wrapper<T8> t8 = def<T8>::f(),
	  wrapper<T9> t9 = def<T9>::f())
      : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9) {}


    template<class U1, class U2>
    tuple(const cons<U1, U2>& p) : base(p) {}

    template <class U1, class U2>
    tuple& operator=(const cons<U1, U2>& k) { 
      base::operator=(k); 
      return *this;
    }

    template <class U1, class U2>
    tuple& operator=(const std::pair<U1, U2>& k) { 
      BOOST_STATIC_ASSERT(length<tuple>::value == 2);// check_length = 2
      this->head = k.first;
      this->tail.head = k.second; 
      return *this;
    }
    
  };
  
  // The empty tuple
  template <>
  class tuple<null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type, null_type>  : 
    public null_type 
  {
  public:
    typedef null_type base;
  };
  
  
}

    /*    tuple(typename access_traits_parameter<T0>::type t0)
      : base(t0, detail::cnull(), detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull()) {}

    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1)
      : base(t0, t1, detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull()) {}

    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2)
      : base(t0, t1, t2, detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull()) {}

    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3)
      : base(t0, t1, t2, t3, detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull(), 
		  detail::cnull()) {}

    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3,
	  typename access_traits_parameter<T4>::type t4)
      : base(t0, t1, t2, t3, t4, detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull(), detail::cnull()) {}
    
    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3,
	  typename access_traits_parameter<T4>::type t4,
	  typename access_traits_parameter<T5>::type t5)
      : base(t0, t1, t2, t3, t4, t5, detail::cnull(), detail::cnull(), 
		  detail::cnull(), detail::cnull()) {}
    
    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3,
	  typename access_traits_parameter<T4>::type t4,
	  typename access_traits_parameter<T5>::type t5,
	  typename access_traits_parameter<T6>::type t6)
      : base(t0, t1, t2, t3, t4, t5, t6, detail::cnull(), 
		  detail::cnull(), detail::cnull()) {}
    
    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3,
	  typename access_traits_parameter<T4>::type t4,
	  typename access_traits_parameter<T5>::type t5,
	  typename access_traits_parameter<T6>::type t6,
	  typename access_traits_parameter<T7>::type t7)
      : base(t0, t1, t2, t3, t4, t5, t6, t7, detail::cnull(), 
		  detail::cnull()) {}

    tuple(typename access_traits_parameter<T0>::type t0,
	  typename access_traits_parameter<T1>::type t1,
	  typename access_traits_parameter<T2>::type t2,
	  typename access_traits_parameter<T3>::type t3,
	  typename access_traits_parameter<T4>::type t4,
	  typename access_traits_parameter<T5>::type t5,
	  typename access_traits_parameter<T6>::type t6,
	  typename access_traits_parameter<T7>::type t7,
	  typename access_traits_parameter<T8>::type t8)
      : base(t0, t1, t2, t3, t4, t5, t6, t7, t8, detail::cnull()) {}

    */
