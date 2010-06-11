project (boost_mtl4_tests)

# DERIVED variables
# none in this CMakeLists.txt

# collect the source files
file (GLOB SOURCES *.cpp)

# collect the include files
file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/mtl/*.hpp)

#define the include directories this library depends on
#headers for cpp's Files
include_directories("${BOOST_ROOT}")           
include_directories("${MTL_BOOST_ROOT}")


# message("CmakeList 2 Boost_root: "${BOOST_ROOT})
# message("CmakeList 2 Mtl_Boost_root: "${MTL_BOOST_ROOT})
# message("CmakeList 2 NVCC_path: "${NVCC_PATH})

###############Cuda´s Things##############
file (GLOB CUSOURCES *.cu)
#file (GLOB INCLUDES ${MTL_BOOST_ROOT}/boost/numeric/mtl/cuda/*.cu)

#INCLUDE(${CMAKE_MODULE_PATH}/cuda/FindCuda.cmake)  
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

# set(EXECUTABLE_OUTPUT_PATH prueba)
# message("aqui---> "${EXECUTABLE_OUTPUT_PATH})


foreach (source ${SOURCES})
    get_filename_component (test ${source} NAME_WE)		# filename without extension
    add_executable (${test} ${source} ${INCLUDES})
    SET(LIST_OF_TEST ${LIST_OF_TEST} ${test})
#    add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
endforeach (source)





# create_test_sourcelist(
#           ${SOURCES} #sourceListName
#           prueba_testing
#           ${LIST_OF_TEST} ${test}
# #          EXTRA_INCLUDE include.h
# #          FUNCTION function
# )




########################################### END MTL4 CPP


# ##############################    cccccc   UUU    UUU  DDDDDD      AA     #################
# ##############################   ccc       UU      UU  DD    DD   A AA    #################
# ##############################   ccc       UU      UU  DD    DD  AAAAAA   #################
# ##############################    cccccc     UUUUUU    DDDDDD   AA    AA  #################




if(NVCC_PATH)
foreach (source ${CUSOURCES})
  get_filename_component (test ${source} NAME_WE) #NAME_WE without extention
  get_filename_component (src ${source} NAME) 
  CUDA_COMPILE(${test} ${src})
endforeach (source)
endif(NVCC_PATH)













