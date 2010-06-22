project (mtl4_itl)

# DERIVED variables
# none in this CMakeLists.txt
# This makefile is meant to be the toplevel for the examples

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT "please-set-mtl4-root-dir" CACHE STRING "BOOST MTL4 root directory")


# collect the source files
file (GLOB SOURCES *.cpp)
# collect the include files so that each example program has a complete browsable MTL class structure
file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/itl/test/*.hpp)

#define the include directories this library depends on
include_directories("${BOOST_ROOT}")
include_directories("${MTL_BOOST_ROOT}")



###############Cuda´s Things##############
file (GLOB CUSOURCES *.cu)

INCLUDE(${MTL4_ROOT_PATH}/FindCuda/FindCuda.cmake) #MTL4´s FindCuda 

#headers for cuda's Files
CUDA_INCLUDE_DIRECTORIES(
  ${MTL4_ROOT_PATH}/
  ${BOOST_ROOT}/
)
#End#######################################





###################################### MTL4 CPP
# .NET 2003
if (MSVC71)
    add_definitions(/DMTL_ASSERT_FOR_THROW /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC71)
# .NET 2005
if (MSVC80)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC80)
# .NET 2008
if (MSVC90)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC90)

# enable testing
enable_testing()

# cycle through the sources
# The directory assumes that each cpp file is a separate test
# so, create a executable target and an associated test target
foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# NAME_WE == fileName Without Extension
    add_executable (${test} ${source} ${INCLUDES})
    add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)
########################################### END MTL4 CPP



# Cuda
if(NVCC_PATH)
foreach (source ${CUSOURCES})
  get_filename_component (test ${source} NAME_WE) #NAME_WE without extention
  get_filename_component (src ${source} NAME) 
  CUDA_COMPILE(${test} ${src})
endforeach (source)
endif(NVCC_PATH)


