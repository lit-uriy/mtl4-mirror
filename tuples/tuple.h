// -*- mode: c++; -*-

#include "boost/type_traits.hpp"
#include <utility> // for pair.hpp


//#include <cstring>


namespace std {

  template <int N> struct index_holder;

  template <int N> void index(const index_holder<N>&) {}

  template <class T>
  void boo2(T& t) {
  }
  void f2() {
    boo2(index<1>);
  }
  
}

template <class T>
void boo(T& t) {
}

#include <cstring>

void f() {
  
  boo(std::index<1>);
}





namespace std {

  using ::boost::is_function; 
  using ::boost::add_reference;
  using ::boost::add_const;
  using ::boost::remove_volatile;

  // -- null_type --------------------------------------------------------
  struct null_type {
    static const int size = 0;
  };

  // - tuple forward declaration -------------------------------------------
  template <
    class T0 = null_type, class T1 = null_type, class T2 = null_type, 
    class T3 = null_type, class T4 = null_type, class T5 = null_type, 
    class T6 = null_type, class T7 = null_type, class T8 = null_type, 
    class T9 = null_type>
  class tuple; 


  namespace tuple_detail {

    inline const null_type c_null_type () { return null_type(); }


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


    template <class T> struct parameter {
      typedef typename add_reference<
        typename add_const<typename remove_volatile<T>::type>::type
      >::type type;
      // when core issue 106 (reference to reference get resolved,
      // and when compilers handle adding const to reference types and types
      // that are already const this can be written as:

      // typedef const typename remove_volatile<T>::type & type;
    };
    
  }









  template<class T>
  struct tuple_size  {
    static const int value = 1 + tuple_size<typename T::tail_type>::value;
  };
  template<>
  struct tuple_size<null_type> {
    static const int value = 0; 
  };
  // this is since the empty tuple inhertis from null_type, so the primary
  // template would match
  template<>
  struct tuple_size<tuple<> > {
    static const int value = 0;
  };





    
  template <class H, class T>
  class cons {

  public:

    typedef H head_type;
    typedef T tail_type;

    static const int size = tail_type::size + 1;

    template <int N>
    struct element {
      typedef 
  private:
    // These are not needed when core issue 106 gets resolved and
    // when adding const to anything works (like it should already)
    typedef typename add_reference<head_type>::type ref_head_type;
    typedef typename add_reference<
      typename add_const<head_type>::type
    >::type  cref_head_type;

    // The proposed definitions are:
    //    typedef stored_head_type& ref_head_type;
    //    typedef const stored_head_type& c_ref_type;

    
    head_type head_; 
    tail_type tail_;

  public:
    // head_type& 
    ref_head_type head() { return head_; }
    // const head_type&
    cref_head_type head() const { return head_; }
      
    tail_type& tail() { return tail_; }
    const tail_type& tail() const { return tail_; }
      
    cons() : head_(), tail_() {}
      
    // add_reference can be replaced by & when core issue 106 gets resolved
    cons(typename tuple_detail::parameter<head_type>::type h,
         const tail_type& t) : head_(h), tail_(t) {}

    template <class H2, class T2>
    cons(const cons<H2, T2>& u ) : head_(u.head), tail_(u.tail) {}

    template <class HT2, class TT2>
    cons& operator=( const cons<HT2, TT2>& u ) { 
      head_ = u.head(); 
      tail_ = u.tail(); 
      return *this; 
    }

    
    // must define assignment operator explicitly, implicit version is 
    // illformed if HT is a reference (12.8. (12))
    cons& operator=(const cons& u) { 
      head_ = u.head_; 
      tail_ = u.tail_;  
      return *this; 
    }
    
    template <class T1, class T2>
    cons& operator=( const std::pair<T1, T2>& u ) { 
      BOOST_STATIC_ASSERT(tuple_size<cons>::value == 2); // check size = 2
      head_ = u.first; 
      tail_.head_ = u.second; 
      return *this;
    }

  protected:
    template <class T1, class T2, class T3, class T4, class T5, 
      class T6, class T7, class T8, class T9, class T10>
    cons( T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, 
	  T6& t6, T7& t7, T8& t8, T9& t9, T10& t10 ) 
      : head_ (t1), 
      tail_ (t2, t3, t4, t5, t6, t7, t8, t9, t10, tuple_detail::c_null_type())
    {}
  };    
    
  template <class H>
  class cons<H, null_type> {

  public:
    static const int size = 1;    
    typedef H head_type;
    typedef null_type tail_type;

  private:
    // These are not needed when core issue 106 gets resolved and
    // when adding const to anything works (like it should already)
    typedef typename add_reference<head_type>::type ref_head_type;
    typedef typename add_reference<
      typename add_const<head_type>::type
    >::type  cref_head_type;

    // The proposed definitions are:
    //    typedef stored_head_type& ref_head_type;
    //    typedef const stored_head_type& c_ref_type;

  public:
    
    typedef H head_type;
    typedef null_type tail_type;

    head_type head_;

    // head_type&
    ref_head_type head() { return head_; }
    // const head_type&
    cref_head_type head() const { return head_; }
    
    const null_type tail() { return null_type(); }
    const null_type tail() const { return null_type(); }      
   

    cons() : head_() {}

    cons(typename tuple_detail::parameter<head_type>::type h,
	 const null_type& = null_type())
      : head_ (h) {}  

    template <class HT2>
    cons( const cons<HT2, null_type>& u ) : head_(u.head()) {}
  
    template <class HT2>
    cons& operator=(const cons<HT2, null_type>& u ) { 
      head_ = u.head(); 
      return *this; 
    }

    // must define assignment operator explicitely, implicit version 
    // is illformed if HT is a reference
    cons& operator=(const cons& u) { head_ = u.head_; return *this; }

    template<class T1>
    cons(T1& t1, const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&, 
	 const null_type&, const null_type&, const null_type&)
      : head_ (t1) {}

  };


  namespace tuple_detail {

    // Tuple to cons mapper --------------------------------------------------
    template <class T0, class T1, class T2, class T3, class T4, 
      class T5, class T6, class T7, class T8, class T9>
    struct tuple_to_cons
    {
      typedef cons<T0, 
	typename tuple_to_cons<T1, T2, T3, T4, T5, 
	                           T6, T7, T8, T9, null_type>::type
      > type;
    };

    // The empty tuple is a null_type
    template <>
    struct tuple_to_cons<null_type, null_type, null_type, null_type, 
                             null_type, null_type, null_type, null_type, 
                             null_type, null_type>
    {
      typedef null_type type;
    };

    // tuple default argument wrappers ---------------------------------------
    // Work only for null_types;

    template <class T> struct generate_error {};
    //    template <class T> T& def(); 
    template <class T> 
    struct def_value {

      static typename add_reference<T>::type 
      f() {
	return generate_error<T>::TOO_FEW_ARGUMENTS_FOR_TUPLE_CONSTRUCTOR;
      }
    };
    template <> 
    struct def_value<null_type> {

      static const null_type
      f() {
	return null_type(); 
      }
    };

  }


  // tuple template
  template <class T0, class T1, class T2, class T3, class T4, 
            class T5, class T6, class T7, class T8, class T9>

  class tuple : 
    public tuple_detail::tuple_to_cons<
      T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
    >::type   
  {
  public:
    typedef typename tuple_detail::tuple_to_cons<
      T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
    >::type base;

    typedef typename base::head_type head_type;
    typedef typename base::tail_type tail_type;  

    tuple() : base() {}
  
    // tuple_traits_parameter<T>::type takes non-reference types as const T& 
    explicit 
    tuple(typename tuple_detail::parameter<T0>::type t0,
	  typename tuple_detail::parameter<T1>::type t1 
	    = tuple_detail::def_value<T1>::f(),
	  typename tuple_detail::parameter<T2>::type t2 
	    = tuple_detail::def_value<T2>::f(),
	  typename tuple_detail::parameter<T3>::type t3 
	    = tuple_detail::def_value<T3>::f(),
	  typename tuple_detail::parameter<T4>::type t4 
	    = tuple_detail::def_value<T4>::f(),
	  typename tuple_detail::parameter<T5>::type t5 
	    = tuple_detail::def_value<T5>::f(),
	  typename tuple_detail::parameter<T6>::type t6 
	    = tuple_detail::def_value<T6>::f(),
	  typename tuple_detail::parameter<T7>::type t7 
	    = tuple_detail::def_value<T7>::f(),
	  typename tuple_detail::parameter<T8>::type t8 
	    = tuple_detail::def_value<T8>::f(),
	  typename tuple_detail::parameter<T9>::type t9 
	    = tuple_detail::def_value<T9>::f())
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
      BOOST_STATIC_ASSERT(tuple_size<tuple>::value == 2);// check_size = 2
      base::operator=(k);
      return *this;
    }
    
  };
  
  // The empty tuple
  template <>
  class tuple<
    null_type, null_type, null_type, null_type, null_type, 
    null_type, null_type, null_type, null_type, null_type> : public null_type {
  public:
    typedef null_type base;
  };



// -cons type accessors ----------------------------------------
// typename tuple_element<N,T>::type gets the type of the 
// Nth element ot T, first element is at index 0
// -------------------------------------------------------

//    template<int N, class T>
//    class tuple_element {
//      typedef typename T::tail_type Next;
//    public:
//      typedef typename tuple_element<N-1, Next>::type type;
//    };


  template<int N, class T>
  class tuple_element : public tuple_element<N-1, typename T::tail_type> {};

  template<class T>
  struct tuple_element<0,T> {
    typedef typename T::head_type type;
  };


// - cons getters --------------------------------------------------------
// called: get_class<N>::get<RETURN_TYPE>(aTuple)
  
  namespace tuple_detail {

    template< int N >
    struct get_class {
      template <class HT, class TT>
      inline static 
      //      const typename tuple_element<N, cons<HT, TT> >::type&
      typename add_reference<
        typename add_const<
          typename tuple_element<N, cons<HT, TT> >::type
        >::type
      >::type
      get(const cons<HT, TT>& t) {
	return get_class<N-1>::template get<RET>(t.get_tail());
      }
      template <class HT, class TT>
      inline static 
      typename add_reference<
          typename tuple_element<N, cons<HT, TT> >::type
      >::type
      //      typename tuple_element<N, cons<HT, TT> >::type&
      get(cons<HT, TT>& t) {
	return get_class<N-1>::template get<RET>(t.get_tail());
      }
    };
    
    template<>
    struct get_class<0> {
      template <class HT, class TT>
      //      const typename tuple_element<N, cons<HT, TT> >::type&
      inline static typename add_reference<typename add_const<HT>::type>::type
      get(const cons<HT, TT>& t) {
	return t.get_head();
      }
      template <class HT, class TT>
      //      typename tuple_element<N, cons<HT, TT> >::type&
      inline static typename add_reference<HT>::type
      get(cons<HT, TT>& t) {
	return t.get_head();
      }
    };
    
  } // tuple_detail


// get function for const cons-lists, returns a const reference to
// the element. If the element is a reference, returns the reference
// as such (that is, can return a non-const reference)

  template<int N, class HT, class TT>
      //      const typename tuple_element<N, cons<HT, TT> >::type&
  inline typename add_reference<
    typename add_const<typename tuple_element<N, cons<HT, TT> >::type>::type
  >::type
  get(const cons<HT, TT>& c) { 
   return tuple_detail::get_class<N>::get(c); 
  } 
 
// get function for non-const cons-lists, returns a reference to the element
  
  template<int N, class HT, class TT>
      //      typename tuple_element<N, cons<HT, TT> >::type&
  inline typename add_reference<
    typename tuple_element<N, cons<HT, TT> >::type
  >::type
  get(cons<HT, TT>& c) { 
   return tuple_detail::get_class<N>::get(c); 
  } 

  // wrapper

  template <class T>
  class generic_holder {
    T data;
    typedef typename add_reference<typename add_const<T>::type>::type par_t;
  public:
    operator T() { return data; }
    T unwrap() { return data; }

    explicit generic_holder(par_t t) : data(t) {}
  };

  template <class T> inline generic_holder<T&> ref(T& t) { 
    return generic_holder<T&>(t); 
  }
  template <class T> inline generic_holder<const T&> cref(const T& t) { 
    return generic_holder<const T&>(t); 
  }
  // make_tuple ---------------------------------------------------------


  namespace tuple_detail {

    template<class T>
    struct arg_traits {
      typedef T type; 
    };

    // The is_function test was there originally for plain function types, 
    // which can't be stored as such (we must either store them as references or
    // pointers). Such a type could be formed if make_tuple was called with a 
    // reference to a function.
    // But this would mean that a const qualified function type was formed in
    // the make_tuple function and hence make_tuple can't take a function
    // reference as a parameter, and thus T can't be a function type.
    // So is_function test was removed.
    // (14.8.3. says that type deduction fails if a cv-qualified function type
    // is created. (It only applies for the case of explicitly specifying template
    // args, though?)) (JJ)

    template<class T>
    struct arg_traits<T&> {
      typedef typename
      tuple_detail::generate_error<T&>::
      ARG_TRAITS_DOES_NOT_SUPPORT_REFERENCE_TYPES error;
    }; 
    
    // Arrays can't be stored as plain types; convert them to references.
    // All arrays are converted to const. This is because make_tuple takes its
    // parameters as const T& and thus the knowledge of the potential 
    // non-constness of actual argument is lost.
    template<class T, int n>  struct arg_traits <T[n]> {
      typedef const T (&type)[n];
    };
    
    template<class T, int n> 
    struct arg_traits<const T[n]> {
      typedef const T (&type)[n];
    };
    
    template<class T, int n>  struct arg_traits<volatile T[n]> {
      typedef const volatile T (&type)[n];
    };
    
    template<class T, int n> 
    struct arg_traits<const volatile T[n]> {
      typedef const volatile T (&type)[n];
    };
    
    template<class T> 
    struct arg_traits<generic_holder<T> >{
      typedef T type;
    };
    
    
    // a helper traits to make the make_tuple functions shorter
    template <
    class T0 = null_type, class T1 = null_type, class T2 = null_type, 
      class T3 = null_type, class T4 = null_type, class T5 = null_type, 
      class T6 = null_type, class T7 = null_type, class T8 = null_type, 
      class T9 = null_type
    >
    struct make_tuple_traits {
      typedef
      tuple<typename arg_traits<T0>::type, 
	typename arg_traits<T1>::type, 
	typename arg_traits<T2>::type, 
	typename arg_traits<T3>::type, 
	typename arg_traits<T4>::type, 
	typename arg_traits<T5>::type, 
	typename arg_traits<T6>::type, 
	typename arg_traits<T7>::type,
	typename arg_traits<T8>::type,
	typename arg_traits<T9>::type> type;
    };
  
  } // end tuple_detail
    
    // -make_tuple function templates -----------------------------------
  inline tuple<> make_tuple() {
    return tuple<>(); 
  }

  template<class T0>
  inline typename tuple_detail::make_tuple_traits<T0>::type
  make_tuple(const T0& t0) {
    typedef typename tuple_detail::make_tuple_traits<T0>::type t;
    return t(t0);
  }
    
  template<class T0, class T1>
  inline typename tuple_detail::make_tuple_traits<T0, T1>::type
  make_tuple(const T0& t0, const T1& t1) {
    typedef typename tuple_detail::make_tuple_traits<T0, T1>::type t;
    return t(t0, t1);
  }
    
  template<class T0, class T1, class T2>
  inline typename tuple_detail::make_tuple_traits<T0, T1, T2>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2) {
    typedef typename tuple_detail::make_tuple_traits<T0, T1, T2>::type t;
    return t(t0, t1, t2);
  }
    
  template<class T0, class T1, class T2, class T3>
  inline typename tuple_detail::make_tuple_traits<T0, T1, T2, T3>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3) {
    typedef typename tuple_detail::make_tuple_traits<T0, T1, T2, T3>::type t;
    return t(t0, t1, t2, t3);
  }
    
  template<class T0, class T1, class T2, class T3, class T4>
  inline typename tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4) {
    typedef typename 
      tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4>::type t;
    return t(t0, t1, t2, t3, t4); 
  }
    
  template<class T0, class T1, class T2, class T3, class T4, class T5>
  inline 
  typename tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4, const T5& t5) {
    typedef typename 
      tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5>::type t;
    return t(t0, t1, t2, t3, t4, t5); 
  }
    
  template<class T0, class T1, class T2, class T3, class T4, class T5, 
    class T6>
  inline 
  typename tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5, T6>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4, const T5& t5, const T6& t6) {
    typedef typename 
      tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5, T6>::type t;
    return t(t0, t1, t2, t3, t4, t5, t6);
  }
    
  template<class T0, class T1, class T2, class T3, class T4, class T5, 
    class T6, class T7>
  inline typename 
  tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5, T6, T7>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4, const T5& t5, const T6& t6, const T7& t7) {
    typedef typename tuple_detail::make_tuple_traits<
      T0, T1, T2, T3, T4, T5, T6, T7
      >::type t;
    return t(t0, t1, t2, t3, t4, t5, t6, t7); 
  }
    
  template<class T0, class T1, class T2, class T3, class T4, class T5, 
    class T6, class T7, class T8>
  inline typename 
  tuple_detail::make_tuple_traits<T0, T1, T2, T3, T4, T5, T6, T7, T8>::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4, const T5& t5, const T6& t6, const T7& t7,
	     const T8& t8) {
    typedef typename tuple_detail::make_tuple_traits<
      T0, T1, T2, T3, T4, T5, T6, T7, T8
      >::type t;
    return t(t0, t1, t2, t3, t4, t5, t6, t7, t8); 
  }
    
  template<class T0, class T1, class T2, class T3, class T4, class T5, 
    class T6, class T7, class T8, class T9>
  inline typename 
  tuple_detail::make_tuple_traits<
  T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
  >::type
  make_tuple(const T0& t0, const T1& t1, const T2& t2, const T3& t3,
	     const T4& t4, const T5& t5, const T6& t6, const T7& t7,
	     const T8& t8, const T9& t9) {
    typedef typename tuple_detail::make_tuple_traits<
      T0, T1, T2, T3, T4, T5, T6, T7, T8, T9
      >::type t;
    return t(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9); 
  }
    
  // Tie function templates -------------------------------------------------
  template<class T1>
  inline tuple<T1&> tie(T1& t1) {
    return tuple<T1&> (t1);
  }
    
  template<class T1, class T2>
  inline tuple<T1&, T2&> tie(T1& t1, T2& t2) {
    return tuple<T1&, T2&> (t1, t2);
  }

  template<class T1, class T2, class T3>
  inline tuple<T1&, T2&, T3&> tie(T1& t1, T2& t2, T3& t3) {
    return tuple<T1&, T2&, T3&> (t1, t2, t3);
  }

  template<class T1, class T2, class T3, class T4>
  inline tuple<T1&, T2&, T3&, T4&> tie(T1& t1, T2& t2, T3& t3, T4& t4) {
    return tuple<T1&, T2&, T3&, T4&> (t1, t2, t3, t4);
  }

  template<class T1, class T2, class T3, class T4, class T5>
  inline tuple<T1&, T2&, T3&, T4&, T5&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5) {
    return tuple<T1&, T2&, T3&, T4&, T5&> (t1, t2, t3, t4, t5);
  }

  template<class T1, class T2, class T3, class T4, class T5, class T6>
  inline tuple<T1&, T2&, T3&, T4&, T5&, T6&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6) {
    return tuple<T1&, T2&, T3&, T4&, T5&, T6&> (t1, t2, t3, t4, t5, t6);
  }

  template<class T1, class T2, class T3, class T4, class T5, class T6, 
    class T7>
  inline tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7) {
    return tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&> 
      (t1, t2, t3, t4, t5, t6, t7);
  }

  template<class T1, class T2, class T3, class T4, class T5, class T6, 
    class T7, class T8>
  inline tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7, T8& t8) {
    return tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&> 
      (t1, t2, t3, t4, t5, t6, t7, t8);
  }

  template<class T1, class T2, class T3, class T4, class T5, class T6, 
    class T7, class T8, class T9>
  inline tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7, T8& t8, 
      T9& t9) {
    return tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&> 
      (t1, t2, t3, t4, t5, t6, t7, t8, t9);
  }

  template<class T1, class T2, class T3, class T4, class T5, class T6, 
    class T7, class T8, class T9, class T10>
  inline tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&, T10&> 
  tie(T1& t1, T2& t2, T3& t3, T4& t4, T5& t5, T6& t6, T7& t7, T8& t8, 
      T9& t9, T10& t10) {
    return tuple<T1&, T2&, T3&, T4&, T5&, T6&, T7&, T8&, T9&, T10&> 
      (t1, t2, t3, t4, t5, t6, t7, t8, t9, t10);
  }

  // comparisons

  inline bool operator==(const null_type&, const null_type&) { return true; }
  inline bool operator>=(const null_type&, const null_type&) { return true; }
  inline bool operator<=(const null_type&, const null_type&) { return true; }
  inline bool operator!=(const null_type&, const null_type&) { return false; }
  inline bool operator<(const null_type&, const null_type&) { return false; }
  inline bool operator>(const null_type&, const null_type&) { return false; }


  namespace tuple_detail {
    // comparison operators check statically the length of its operands and
    // delegate the comparing task to the following functions. Hence
    // the static check is only made once (should help the compiler).  
    // These functions assume tuples to be of the same length.

    template<class T1, class T2>
    inline bool eq(const T1& lhs, const T2& rhs) {
      return lhs.get_head() == rhs.get_head() &&
	eq(lhs.get_tail(), rhs.get_tail());
    }
    template<>
    inline bool eq<null_type,null_type>(const null_type&, const null_type&) { 
      return true; 
    }

    template<class T1, class T2>
    inline bool neq(const T1& lhs, const T2& rhs) {
      return lhs.get_head() != rhs.get_head()  ||
	neq(lhs.get_tail(), rhs.get_tail());
    }
    template<>
    inline bool neq<null_type,null_type>(const null_type&, const null_type&) { 
      return false; 
    }

    template<class T1, class T2>
    inline bool lt(const T1& lhs, const T2& rhs) {
      return lhs.get_head() < rhs.get_head()  ||
	!(rhs.get_head() < lhs.get_head()) &&
	lt(lhs.get_tail(), rhs.get_tail());
    }
    template<>
    inline bool lt<null_type,null_type>(const null_type&, const null_type&) { 
      return false; 
    }

    template<class T1, class T2>
    inline bool gt(const T1& lhs, const T2& rhs) {
      return lhs.get_head() > rhs.get_head()  ||
	!(rhs.get_head() > lhs.get_head()) &&
	gt(lhs.get_tail(), rhs.get_tail());
    }
    template<>
    inline bool gt<null_type,null_type>(const null_type&, const null_type&) { 
      return false; 
    }

    template<class T1, class T2>
    inline bool lte(const T1& lhs, const T2& rhs) {
      return lhs.get_head() <= rhs.get_head()  &&
	( !(rhs.get_head() <= lhs.get_head()) ||
	  lte(lhs.get_tail(), rhs.get_tail()));
    }
    template<>
    inline bool lte<null_type,null_type>(const null_type&, const null_type&) {
      return true; 
    }

    template<class T1, class T2>
    inline bool gte(const T1& lhs, const T2& rhs) {
      return lhs.get_head() >= rhs.get_head()  &&
	( !(rhs.get_head() >= lhs.get_head()) ||
	  gte(lhs.get_tail(), rhs.get_tail()));
    }
    template<>
    inline bool gte<null_type,null_type>(const null_type&, const null_type&) {
      return true; 
    }

  } // end of namespace tuple_detail


  // equal ----

  template<class T1, class T2, class S1, class S2>
  inline bool operator==(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {
    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return  tuple_detail::eq(lhs, rhs);
  }

  // not equal -----

  template<class T1, class T2, class S1, class S2>
  inline bool operator!=(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {

    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return tuple_detail::neq(lhs, rhs);
  }

  // <
  template<class T1, class T2, class S1, class S2>
  inline bool operator<(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {
    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return tuple_detail::lt(lhs, rhs);
  }

  // >
  template<class T1, class T2, class S1, class S2>
  inline bool operator>(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {
    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return tuple_detail::gt(lhs, rhs);
  }

  // <=
  template<class T1, class T2, class S1, class S2>
  inline bool operator<=(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {
    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return tuple_detail::lte(lhs, rhs);
  }

  // >=
  template<class T1, class T2, class S1, class S2>
  inline bool operator>=(const cons<T1, T2>& lhs, const cons<S1, S2>& rhs)
  {
    // check that tuple tuple_sizes are equal
    BOOST_STATIC_ASSERT(tuple_size<T2>::value == tuple_size<S2>::value);

    return tuple_detail::gte(lhs, rhs);
  }



}















