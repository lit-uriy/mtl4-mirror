// $COPYRIGHT$

#ifndef MTL_MTL_EXCEPTION_INCLUDE
#define MTL_MTL_EXCEPTION_INCLUDE

#include <cassert>

namespace mtl {

// If MTL_ASSERT_FOR_THROW is defined all throws become assert
// MTL_DEBUG_THROW_IF completely disappears if NDEBUG is defined
#ifndef NDEBUG
#  ifdef MTL_ASSERT_FOR_THROW
#    define MTL_DEBUG_THROW_IF(Test, Exception) \
     {                                          \
        assert(Test)                            \
     }
#  else
#    define MTL_DEBUG_THROW_IF(Test, Exception) \
     {                                          \
        if (Test) throw Exception;              \
     }
#  endif
#else
#  define MTL_DEBUG_THROW_IF(Test,Exception)
#endif


#ifdef MTL_ASSERT_FOR_THROW
#  define MTL_THROW_IF(Test, Exception)       \
   {                                          \
      assert(Test)                            \
   }
#else
#  define MTL_THROW_IF(Test, Exception)       \
   {                                          \
      if (Test) throw Exception;              \
   }
#endif

    struct bad_range {};

} // namespace mtl

#endif // MTL_MTL_EXCEPTION_INCLUDE
