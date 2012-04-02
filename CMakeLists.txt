project (mtl4)

cmake_minimum_required(VERSION 2.8)
option(ENABLE_GCOV "enable gnu coverage flags for gnu compiler" OFF)
#look for mtl4, use THIS directory as mtl dir
find_package(MTL REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MTL_INCLUDE_DIRS})

#REMAINING PART USES MTL4
include(CheckCXXCompilerFlag)

#let cmake look for the svn-version, only needed for packages?
if (NOT MSVC)
	find_package(Subversion)
	if(Subversion_FOUND)
		Subversion_WC_INFO(${CMAKE_CURRENT_SOURCE_DIR} mtlSubinfo)
		set(CurrentRevision ${mtlSubinfo_WC_REVISION})
#	message("current revision: ${mtlSubinfo_WC_REVISION}")
	else(Subversion_FOUND)
		set(CurrentRevision "0")
	endif(Subversion_FOUND)
endif (NOT MSVC)

configure_file(boost/numeric/mtl/version.hpp.in boost/numeric/mtl/version.hpp @ONLY)
add_definitions("-DMTL_ASSERT_FOR_THROW")

#look for compiler specific settings
if (MSVC)
	# to quiet DevStudio 2005 secure code warnings
    add_definitions(/D_SCL_SECURE_NO_WARNINGS)
    add_definitions(/D_CRT_SECURE_NO_WARNINGS)
    add_definitions(/DMTL_ASSERT_FOR_THROW /D_CRT_SECURE_NO_DEPRECATE /DNOMINMAX /D_CONSOLE /D"_HAS_ITERATOR_DEBUGGING=0" /D"_SECURE_SCL=0" )
	# specialty flags such as the C++ language exception model etc. come from the init flags determined by CMAKE
	# you may override them, but I have not found it necessary
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE_INIT} /wd4018 /wd4099 /wd4522 /wd4996 /wd4355 /wd4244 /Z7")
endif (MSVC)

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions("-Wall -Wextra -Wpointer-arith -Wcast-align -Wcast-qual -Wwrite-strings")
  check_cxx_compiler_flag("-Wno-long-long" WITH_NO_LONG_LONG)
  if (${WITH_NO_LONG_LONG})
    message(STATUS "Add -Wno-long-long")
    add_definitions("-Wno-long-long")
  endif()
  #vampir does not support pedantic..
  if(NOT HAVE_VAMPIR)
    add_definitions("-pedantic")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux") # skip Werror on Darwin for the moment
      add_definitions("-Werror")
    endif() 
  endif()
endif()

set(BUILDNAME "${TESTBUILDNAME}")
enable_testing()
include(CTest)
add_subdirectory(libs)	
