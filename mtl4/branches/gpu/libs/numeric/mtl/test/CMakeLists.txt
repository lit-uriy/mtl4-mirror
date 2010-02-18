CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
#cmake_minimum_required(VERSION 2.8)

project (mtl4-gpu_tests)

# DERIVED variables
# none in this CMakeLists.txt

# collect the source files
file (GLOB SOURCES *.cpp)
#file (GLOB SOURCES *.cu)

# collect the include files
file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/mtl/*.hpp)

#define the include directories this library depends on
include_directories("${BOOST_ROOT}")
include_directories("${MTL_BOOST_ROOT}")
include_directories("${CMAKE_SOURCE_DIR}")

#Adding FindCUDA
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/share/cmake-2.8/Modules/FindCUDA")
FIND_PACKAGE(CUDA REQUIRED)

INCLUDE_DIRECTORIES(
  ${CUDA_TOOLKIT_INCLUDE}
#  ${CUDA_CUT_INCLUDE_DIR}
  )


# .NET 2003
if (MSVC71)
    add_definitions(/DMTL_ASSERT_FOR_THROW /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "/EHsc /EHa /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC71)
# .NET 2005
if (MSVC80)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "/EHsc /EHa /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC80)

# cycle through the sources
# The directory assumes that each cpp file is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# filename without extension
    add_executable (${test} ${source} ${INCLUDES})
    add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)


