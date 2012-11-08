project (gpu_mtl4_tests)

file (GLOB CUSOURCES *.cu)

if(NVCC_PATH)
  foreach (source ${CUSOURCES})
    get_filename_component (test ${source} NAME_WE) #NAME_WE without extention
    get_filename_component (src ${source} NAME)
    CUDA_ADD_EXECUTABLE(${test} ${source} OPTIONS ${CUDA_FLAGS})
    #  add_test (${test} ${EXECUTABLE_OUTPUT_PATH}/${test})
  endforeach (source)
else()
  message("Need CUDA Compiler here!!!")
endif()
