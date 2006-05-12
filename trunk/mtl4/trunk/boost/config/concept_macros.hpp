// $COPYRIGHT$

#ifndef  CONCEPT_MACROS_INCLUDE
#define  CONCEPT_MACROS_INCLUDE

#ifdef __GXX_CONCEPTS__
#  define LA_WITH_CONCEPTS
#  define MTL_WITH_CONCEPTS
#  define LA_WHERE(...) where __VA_ARGS__
#  define MTL_WHERE(...) where __VA_ARGS__
#else
#  define LA_NO_CONCEPTS
#  define MTL_NO_CONCEPTS
#  define LA_WHERE(...)
#  define MTL_WHERE(...)
#endif





#endif //  CONCEPT_MACROS_INCLUDE
