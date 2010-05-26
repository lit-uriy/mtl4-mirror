#######FindCuda for MTL4#########
#
#
#
#
##################################





########################################################
# Add include directories to pass to the nvcc command. #
########################################################
MACRO(CUDA_INCLUDE_DIRECTORIES)
  FOREACH(dir ${ARGN})
    SET(CUDA_NVCC_INCLUDE_ARGS ${CUDA_NVCC_INCLUDE_ARGS} -I${dir})
  ENDFOREACH(dir ${ARGN})
ENDMACRO(CUDA_INCLUDE_DIRECTORIES)




########################################################
# Add include command to compile NVCC                  #
########################################################
set(CUDA_NVCC_COMMANDS 
--gpu-architecture sm_13
-DBOOST_NO_INCLASS_MEMBER_INITIALIZATION
-DMTL_SHORT_PRINT
#-O1
#-O2
-O3
#--keep
#-cuda
)




########################################################
# Macro to compile cuda daten.                         #
########################################################
MACRO(CUDA_COMPILE TARGNAME)
add_custom_command(OUTPUT ${TARGNAME}  DEPENDS ${ARGN}  COMMAND nvcc -o ${TARGNAME} ${ARGN} ARGS ${CUDA_NVCC_INCLUDE_ARGS} ${CUDA_NVCC_COMMANDS})
add_custom_target(${TARGNAME}_helper ALL DEPENDS ${TARGNAME} )
ENDMACRO(CUDA_COMPILE)



