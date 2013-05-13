include(CheckCXXCompilerFlag)

# Compiler flag for C++11: special case for VC only, everything else should be equal
# Might need later adaption, once compilers don't support this flag any longer
if(MSVC)
  set(CXX_ELEVEN_FLAG "/Qstd=c++0x")
else()
  set(CXX_ELEVEN_FLAG "-std=c++0x")
endif()

# only performed once (to reevaluated delete CMakeCache.txt)
check_cxx_compiler_flag("${CXX_ELEVEN_FLAG}" CXX_ELEVEN_FLAG_SUPPORTED) 

if (NOT CXX_ELEVEN_FLAG_SUPPORTED)
  if (ENABLE_CXX_ELEVEN)
    message("C++11 flag not supported by your compiler (probably too old).")
  endif()
  return()
endif()

if (NOT ENABLE_CXX_ELEVEN)
  return()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CXX_ELEVEN_FLAG "${CXX_ELEVEN_FLAG} -stdlib=libc++")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lc++")
endif()

message(STATUS "Add ${CXX_ELEVEN_FLAG}")
add_definitions("${CXX_ELEVEN_FLAG}")

set (CXX_ELEVEN_FEATURE_LIST "MOVE" "INITLIST" "STATICASSERT" "AUTO" "RANGEDFOR" "DEFAULTIMPL")

foreach (feature ${CXX_ELEVEN_FEATURE_LIST})
   try_compile(${feature}_RESULT ${CMAKE_BINARY_DIR} "${CMAKE_SOURCE_DIR}/tools/cmake/${feature}_CHECK.cpp" COMPILE_DEFINITIONS "${CXX_ELEVEN_FLAG}")
  # try_compile(${feature}_RESULT . "./${feature}_CHECK.cpp")
   message(STATUS "Support C++11's ${feature} - ${${feature}_RESULT}")
   if (${feature}_RESULT)
     add_definitions("-DMTL_WITH_${feature}")
   endif()
endforeach()