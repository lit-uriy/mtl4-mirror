    template <class T> struct non_storeable_type {
      typedef T type;
      //      typedef typename ct_if<
      //        is_function<T>::value, non_storeable_type<T>, T
      //      >::type type;
    };

    //    template <> struct non_storeable_type<void> {
    //      typedef non_storeable_type<void> type; 
    //    };

    // used as the tuple constructors parameter types 
    // (and in cons consturctors)
    // Rationale: non-reference tuple element types can be cv-qualified.
    // It should be possible to initialize such types with temporaries,
    // and when binding temporaries to references, the reference must
    // be non-volatile and const. 8.5.3. (5)
