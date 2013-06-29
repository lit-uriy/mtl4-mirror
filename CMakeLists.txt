project (mtl4) # GPU version 

cmake_minimum_required(VERSION 2.8)
option(ENABLE_GCOV "enable gnu coverage flags for gnu compiler" OFF)
option(ENABLE_WARNINGS "enable gnu specific warnings" ON)
option(ENABLE_DEVELOPMENT "enable program parts under development" OFF)

#look for mtl4, use THIS directory as mtl dir
find_package(MTL REQUIRED HINTS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${MTL_INCLUDE_DIRS})

# adding the Path of our own Modules
SET(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${MTL4_ROOT_PATH}/Cmake_Module)
SET(MTL4_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})


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

# if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#   set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lstdc++")
# endif()

configure_file(boost/numeric/mtl/version.hpp.in boost/numeric/mtl/version.hpp @ONLY)

if (USE_ASSERTS)
  add_definitions("-DMTL_ASSERT_FOR_THROW")
endif()

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

if( "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND ENABLE_WARNINGS)
  add_definitions("-Wall -Wextra -Wpointer-arith -Wcast-align -Wcast-qual -Wwrite-strings")

  # Normalize locale during test compilation.
  # set(_CheckCXXCompilerFlag_LOCALE_VARS LC_ALL LC_MESSAGES LANG)
  # message ("Flags ${_CheckCXXCompilerFlag_LOCALE_VARS}")
  # foreach(v ${_CheckCXXCompilerFlag_LOCALE_VARS})
  #   set(_CheckCXXCompilerFlag_SAVED_${v} "$ENV{${v}}")
  #   set(ENV{${v}} C)
  # endforeach()

  check_cxx_compiler_flag("-Wno-long-long" WITH_NO_LONG_LONG)
  if (WITH_NO_LONG_LONG)
    message(STATUS "Add -Wno-long-long")
    add_definitions("-Wno-long-long")
  endif()

  check_cxx_compiler_flag("-Wno-unused-local-typedefs" WITH_NO_UNUSED_LOCAL_TYPEDEFS)
  if (WITH_NO_UNUSED_LOCAL_TYPEDEFS)
    message(STATUS "Add -Wno-unused-local-typedefs")
    add_definitions("-Wno-unused-local-typedefs")
  endif()

  # foreach(v ${_CheckCXXCompilerFlag_LOCALE_VARS})
  #   set(ENV{${v}} ${_CheckCXXCompilerFlag_SAVED_${v}})
  #   unset(_CheckCXXCompilerFlag_SAVED_${v})
  # endforeach()
  # unset(_CheckCXXCompilerFlag_LOCALE_VARS)
  
  # vampir does not support pedantic.. Boost generates too many warnings with nvcc anyway, so drop it entirely
  # if(NOT HAVE_VAMPIR)
  #   add_definitions("-pedantic")
  #   if(CMAKE_SYSTEM_NAME STREQUAL "Linux") # skip Werror on Darwin for the moment
  #     add_definitions("-Werror")
  #   endif() 
  # endif()
endif()

if (ENABLE_DEVELOPMENT)
  add_definitions("-DMTL_WITH_DEVELOPMENT")
endif()

if (ENABLE_SHORT_ELE_PROD)
  add_definitions("-DMTL_WITH_SHORT_ELE_PROD")
endif()

set(BUILDNAME "${TESTBUILDNAME}")
enable_testing()
include(CTest)
add_subdirectory(libs)	
