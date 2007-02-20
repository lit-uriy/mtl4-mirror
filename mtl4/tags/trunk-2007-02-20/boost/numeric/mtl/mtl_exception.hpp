// $COPYRIGHT$

#ifndef MTL_MTL_EXCEPTION_INCLUDE
#define MTL_MTL_EXCEPTION_INCLUDE

namespace mtl {

#ifndef NDEBUG
#  define throw_debug_exception(Test,Message) \
   {                                          \
      if (Test) throw Message;                \
   }
#else
#  define throw_debug_exception(Test,Message)
#endif

struct test_exception {};

} // namespace mtl

#endif // MTL_MTL_EXCEPTION_INCLUDE
