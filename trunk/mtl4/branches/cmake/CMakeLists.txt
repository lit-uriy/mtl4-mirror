cmake_minimum_required (VERSION 2.4)

project (boost_mtl4)

# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4

# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
# set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")
set (BOOST_ROOT $ENV{BOOST_ROOT})
find_path(Boost_INCLUDE_DIR boost/version.hpp PATHS "${BOOST_ROOT}")
if(NOT Boost_INCLUDE_DIR)
  message(FATAL_ERROR "boost wasn't found, please set BOOST_ROOT")
endif(NOT Boost_INCLUDE_DIR)

# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

set (MTL_HAS_UMF $ENV{UMF_ROOT})
message(STATUS ${MTL_HAS_UMF})
find_path(DIR_UMF UMFPACK/Source/umf_version.h PATHS "${UMF_ROOT}")
if(NOT DIR_UMF)
  message(STATUS "UMFPack wasn't found, please set UMF_ROOT as Environment Variable if it is installed.")
  set (MTL_HAS_UMF "NOPATH")
endif(NOT DIR_UMF)



# include_directories(${Boost_INCLUDE_DIR})

######################## DevStudio funkyness   ################################
# to quiet DevStudio secure code warnings

# .NET 2005 and .NET 2008
if (MSVC80 OR MSVC90)
	 add_definitions(/D_SCL_SECURE_NO_WARNINGS)
	 add_definitions(/D_CRT_SECURE_NO_WARNINGS)
	 add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_SECURE_SCL=0" )
endif (MSVC80 OR MSVC90)

# .NET 2005
if (MSVC80)
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4535 /wd4554 /wd4244 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4535 /wd4554 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC80)

# .NET 2008
if (MSVC90)
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /EHa /wd4535 /wd4554 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /EHa /wd4535 /wd4554 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
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