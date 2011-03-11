project (mtl4)

cmake_minimum_required(VERSION 2.6)

include(CheckCXXCompilerFlag)

#let cmake look for the svn-version
if (NOT MSVC)
find_package(Subversion)
if(Subversion_FOUND)
	Subversion_WC_INFO(${CMAKE_CURRENT_BINARY_DIR} mtlSubinfo)
	set(CurrentRevision ${mtlSubinfo_WC_REVISION})
#	message("current revision: ${mtlSubinfo_WC_REVISION}")
else(Subversion_FOUND)
	set(CurrentRevision "0")
endif(Subversion_FOUND)
endif (NOT MSVC)
configure_file(boost/numeric/mtl/version.hpp.in boost/numeric/mtl/version.hpp @ONLY)
# DERIVED VARIABLES
# no derived variables, this makefile is the toplevel makefile for boost mtl4


# The BOOST_INCLUDE_DIR needs to point to the release BOOST tree
# so that we can pick up boost/test/minimal.hpp
# set (BOOST_ROOT "please-set-boost-include-dir" CACHE STRING "BOOST include directory")

################ LINUX
#Find Boost  on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(STATUS "You use : "${CMAKE_SYSTEM_NAME})   #system name
  
  #Path of Boost
  set (BOOST_ROOT $ENV{BOOST_ROOT} CACHE PATH "BOOST include directory")
  find_package(Boost)

  if(Boost_FOUND)
   message(STATUS "Boost in " ${Boost_INCLUDE_DIR}) #directory of Boost
   set (BOOST_ROOT ${Boost_INCLUDE_DIR})
  endif(Boost_FOUND)

  #Find Manualy the path of Boost
  if(NOT Boost_FOUND)
    if(NOT IS_DIRECTORY /usr/include/boost)
      message(FATAL_ERROR "Searching Manually Boost: FATAL ERROR: Boost Not Found in /usr/include/boost, Please set BOOST_ROOT")
    else(NOT IS_DIRECTORY /usr/include/boost)
      set (BOOST_ROOT /usr/include/boost/)
      message(STATUS "Searching Manually Boost: Boost Found in /usr/include/boost/")
      find_package(Boost)
    endif(NOT IS_DIRECTORY /usr/include/boost)
  endif(NOT Boost_FOUND)

endif(CMAKE_SYSTEM_NAME STREQUAL "Linux")

################ END LINUX

################ WINDOWS

#Find Boost  on Windows
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
 message(STATUS ${CMAKE_SYSTEM_NAME})
 
  #Path of Boost
  set (BOOST_ROOT $ENV{BOOST_ROOT} CACHE PATH "BOOST include directory")
  find_package(Boost)
  
  if(Boost_FOUND)
   message(STATUS "Boost in " ${Boost_INCLUDE_DIR}) #directory of Boost
   set (BOOST_ROOT ${Boost_INCLUDE_DIR})
  endif(Boost_FOUND)

    #Find Manualy the path of Boost
    if(NOT Boost_FOUND)
      if(NOT IS_DIRECTORY c:/boost)
        message(FATAL_ERROR "Searching Manually Boost: FATAL ERROR : Boost Not Found in c:/boost, Please set BOOST_ROOT")
      else(NOT IS_DIRECTORY c:/boost)
        set (BOOST_ROOT c:/boost)
        message(STATUS "Searching Manually Boost: Found in c:/boost")
	find_package(Boost)
      endif(NOT IS_DIRECTORY c:/boost)
    endif(NOT Boost_FOUND)
endif(CMAKE_SYSTEM_NAME STREQUAL "Windows")

################ END WINDOWS



# set the local BOOST MTL4 include directory
# to pick up the MTL4 include files
set (MTL_BOOST_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "BOOST MTL4 root directory")

# include_directories(${Boost_INCLUDE_DIR})
include_directories(${BOOST_ROOT} ${MTL_BOOST_ROOT})

# Experimental !!!!
# add_definitions("-I/home/pgottsch/Software/arprec-2.2.7/include -DMTL_HAS_ARPREC")
# add_definitions("-static -L/home/pgottsch/Software/arprec-2.2.7/src -larprec")


#################################################################################
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

#################################################################################

check_cxx_compiler_flag("-Wno_long_long" WITH_NO_LONG_LONG)
message ("no long long: ${WITH_NO_LONG_LONG}")
  

find_program(VALGRIND_COMMAND valgrind)
if(VALGRIND_COMMAND)
  set(CTEST_MEMORYCHECK_COMMAND ${VALGRIND_COMMAND})
endif(VALGRIND_COMMAND)

macro (add_macro_flags)
  foreach (flag ${ARGV})
    if (win32)
      add_definitions("/D${flag}")
    else()
      add_definitions("-D${flag}")
    endif()
  endforeach()
endmacro (add_macro_flags)

#add_definitions("-g")
add_macro_flags("MTL_ASSERT_FOR_THROW")

option(ENABLE_GCOV "enable gnu coverage flags for gnu compiler" OFF)
if( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  add_definitions("-Werror -pedantic -Wall -Wextra -Wpointer-arith -Wcast-align -Wcast-qual -Wwrite-strings")
  if(ENABLE_GCOV)
	add_definitions("-fprofile-arcs -ftest-coverage")
  endif(ENABLE_GCOV)
endif()


macro (compile_all testing)
    # cycle through the sources
    # The directory assumes that each cpp file is a separate test
    # so, create a executable target and an associated test target
    foreach (source ${ARGN})
        get_filename_component (test ${source} NAME_WE)	
        add_executable (${test} ${source} ${INCLUDES})
	if(ENABLE_GCOV)
	  target_link_libraries(${test} gcov)
	endif(ENABLE_GCOV)
	# add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
	if (${testing} STREQUAL "true")
	  # message("testing: ${testing} ${source}")
	  add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
	endif()
    endforeach (source)
endmacro (compile_all)

# enable test generation
set(BUILDNAME "${TESTBUILDNAME}")
enable_testing()
include(CTest)


# Descent into the test program directory to build the test suite
# MTL4 test programs
add_subdirectory("libs/numeric/mtl/test")
# as well as the example programs
add_subdirectory("libs/numeric/mtl/examples")
# Experimental and timing are not compiled by default
add_subdirectory("libs/numeric/mtl/experimental" EXCLUDE_FROM_ALL)
add_subdirectory("libs/numeric/mtl/timing" EXCLUDE_FROM_ALL)

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





