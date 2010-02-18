project (boost_mtl4-gpu)

CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
# set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

#set (BOOST_ROOT $ENV{BOOST_ROOT} CACHE PATH "BOOST include directory")
#find_path(Boost_INCLUDE_DIR boost/version.hpp PATHS "${BOOST_ROOT}")
#if(NOT Boost_INCLUDE_DIR)
#  message(FATAL_ERROR "boost wasn't found, please set BOOST_ROOT")
#endif(NOT Boost_INCLUDE_DIR)

set (BOOST_ROOT ENV{BOOST_ROOT} CACHE PATH "BOOST include directory")


#adding FindCuda
SET(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/share/cmake-2.8/Modules/FindCUDA")
FIND_PACKAGE(CUDA REQUIRED)



# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

# include_directories(${Boost_INCLUDE_DIR})
include_directories($ENV{BOOST_ROOT} ${MTL_BOOST_ROOT})

######################## DevStudio funkyness   ################################
# .NET 2003
if (MSVC71)
    add_definitions(/DMTL_ASSERT_FOR_THROW /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" )
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
endif (MSVC71)
# .NET 2005
if (MSVC80)
	# to quiet DevStudio 2005 secure code warnings
    add_definitions(/D_SCL_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC80)

# .NET 2008
if (MSVC90)
	# to quiet DevStudio 2005 secure code warnings
    add_definitions(/D_SCL_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /EHa /wd4554 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /EHa /wd4554 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC90)

# enable test generation
enable_testing()

# Descent into the test program directory to build the test suite
# MTL4 test programs
add_subdirectory("libs/numeric/mtl/test")
# as well as the example programs
add_subdirectory("libs/numeric/mtl/examples")
# ITL test programs
add_subdirectory("libs/numeric/itl/test")

#################################################################################
## handy for learning CMAKE and debugging possible problems
######################## record all the variables CMAKE defined #################
file (WRITE ${CMAKE_CURRENT_BINARY_DIR}/AllVariables.txt "")
get_cmake_property(VARS VARIABLES)
foreach (var ${VARS})
    file (APPEND ${CMAKE_CURRENT_BINARY_DIR}/AllVariables.txt "${var} \"${${var}}\"\n")
endforeach (var ${VARS})
