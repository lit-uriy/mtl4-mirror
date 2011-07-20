
option(ENABLE_VAMPIR "set to true if you wants to use vampir" FALSE)

if(ENABLE_VAMPIR)
	set(VPT_TESTDIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/vpt_test)
	file(WRITE ${VPT_TESTDIR}/CMakeLists.txt "project(vpt_test)\n add_executable(vpt vpt.cc)")
	file(WRITE ${VPT_TESTDIR}/vpt.cc "#include <vt_user.h>\n int main() { VT_USER_START(\"test\"); VT_USER_END(\"test\"); return 0; }")
	try_compile(VPT_TEST ${VPT_TESTDIR} ${VPT_TESTDIR} "vpt_test" OUTPUT_VARIABLE vpt_outvar)
	#message("output: ${vpt_outvar}")
	if(VPT_TEST)
		set(HAVE_VAMPIR_CXX ON)
	else()
		find_program(VAMPIR_CXX vtc++)
		if(VAMPIR_CXX)
			message(FATAL_ERROR "please set the compiler to the vampir compiler \n ${VAMPIR_CXX} \n in cmake") #or use the environmentvariables CC & CXX
		else()
			message(FATAL_ERROR "could not even find the vampir compiler. If it resides in a non-standard path, set the compiler to your vampir compiler")
		endif()
	endif()
#  if(NOT win32)
#    find_program(VAMPIR_CXX vtc++)
#    if(VAMPIR_CXX)
#      include(CMakeForceCompiler)
#      CMAKE_FORCE_CXX_COMPILER(${VAMPIR_CXX} vampir-cxx)
#    endif()
#   find_program(VAMPIR_CC vtcc)
#    if(VAMPIR_CC)
#      include(CMakeForceCompiler)
#      CMAKE_FORCE_C_COMPILER(${VAMPIR_CC} vampir-cc)
#    endif()
#    if(NOT VAMPIR_CXX AND NOT VAMPIR_CC)
##	message(FATAL_ERROR "please set the vampir compiler paths")
    
#    endif()
#  else()


  #Windows
#  endif()
endif(ENABLE_VAMPIR)




#End Looking for Vampir

