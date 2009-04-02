project (boost_mtl4)

cmake_minimum_required(VERSION 2.6)
# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4

if(WIN32)
  set(Boost_USE_STATIC_LIBS ON)
endif(WIN32)

find_package(Boost)
if ( Boost_FOUND )
  include_directories(${Boost_FOUND})
#  link_directories(${Boost_LIBRARY_DIRS})
  add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})
  # Remember to include ${Boost_LIBRARIES} in the target_link_libraries() statement
endif ( Boost_FOUND )
if(NOT Boost_INCLUDE_DIR)
  message(FATAL_ERROR "boost wasn't found, please set BOOST_ROOT")
endif(NOT Boost_INCLUDE_DIR)
#message("Boost include dirs: ${Boost_INCLUDE_DIRS} ")

find_package(Umfpack QUIET)
if ( Umfpack_FOUND )
  include_directories(${Umfpack_FOUND})
endif ( Umfpack_FOUND )
if(NOT Umfpack_FOUND)
  set (UMF_DIR $ENV{UMFPACK_ROOT})
  find_path(UMF_DIR UMFPACK/Source/umf_version.h PATHS "${UMFPACK_ROOT}")
  if(UMF_DIR)
    include_directories(${UMF_DIR})
    message(STATUS "UMFPack found, and included.")
  else(UMF_DIR)
    message(STATUS "UMFPack wasn't found, please set UMFPACK_ROOT as Environment Variable if it is installed.")
  endif(UMF_DIR)
endif(NOT Umfpack_FOUND)

find_package(Lapack QUIET)
if ( Lapack_FOUND )
  include_directories(${Lapack_FOUND})
endif ( Lapack_FOUND )
if(NOT Lapack_FOUND)
  set (LAPack_DIR $ENV{LAPACK_ROOT})
  find_path(LAPack_DIR SRC/chptrf.f PATHS "${LAPACK_ROOT}")
  if(LAPack_DIR)
    include_directories(${LAPack_DIR})
    message(STATUS "LAPack found, and included.")
  else(LAPack_DIR)
    message(STATUS "LAPack wasn't found, please set LAPACK_ROOT as Environment Variable if it is installed.")
  endif(LAPack_DIR)
endif(NOT Lapack_FOUND)

find_package(Blas QUIET)
if ( Blas_FOUND )
  include_directories(${Blas_FOUND})
endif ( Blas_FOUND )
if(NOT Blas_FOUND)
  set (Blas_DIR $ENV{BLAS_ROOT})
  find_path(Blas_DIR cgemm.f PATHS "${BLAS_ROOT}")
  if(Blas_DIR)
    include_directories(${Blas_DIR})
    message(STATUS "Blas found, and included.")
  else(Blas_DIR)
    message(STATUS "Blas wasn't found, please set BLAS_ROOT as Environment Variable if it is installed.")
  endif(Blas_DIR)
endif(NOT Blas_FOUND)

# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

# include_directories(${Boost_INCLUDE_DIR})

######################## DevStudio funkyness   ################################

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
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4535 /wd4554 /wd4244 /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC80)

# .NET 2008
if (MSVC90)
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





