
option(ENABLE_VAMPIR "set to true if you wants to use vampir" FALSE)

if(ENABLE_VAMPIR)
  if(NOT win32)
    find_program(VAMPIR_CXX vtc++)
    if(VAMPIR_CXX)
      include(CMakeForceCompiler)
      CMAKE_FORCE_CXX_COMPILER(${VAMPIR_CXX} vampir-cxx)
    endif()
    find_program(VAMPIR_CC vtcc)
    if(VAMPIR_CC)
      include(CMakeForceCompiler)
      CMAKE_FORCE_C_COMPILER(${VAMPIR_CC} vampir-cc)
    endif()
    if(NOT VAMPIR_CXX AND NOT VAMPIR_CC)
	message(FATAL_ERROR "please set the vampir compiler paths")
    
    endif()
  else()


  #Windows
  endif()
endif(ENABLE_VAMPIR)




#End Looking for Vampir

