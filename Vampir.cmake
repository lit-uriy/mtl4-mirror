find_program(VAMPIR_CXX vtc++)
option(ENABLE_VAMPIR "enable or disable vampir trace" OFF)
if(VAMPIR_CXX)
	set(VAMPIR_FOUND TRUE)
#request only compile arguments
	execute_process(COMMAND ${VAMPIR_CXX} "-vt:show" "-vt:inst" "manual" "-c" RESULT_VARIABLE VT_RES OUTPUT_VARIABLE VT_OUT)
#	message("vt_res: ${VT_RES}")
#	message("vt_out: ${VT_OUT}")
	string(REPLACE " " ";" asList ${VT_OUT} )
	list(LENGTH asList length)
#	message("asList: ${asList}")
#	message("length: ${length}")
	list(GET asList 0 VT_COMPILER)
#remove the compiler
	list(REMOVE_AT asList 0)
	#add each flag except the last to the vampir compile flags
	set(VT_COMPILE_FLAGS )
	while(length GREATER 1)
		list(GET asList 0 CURFLAG)
		list(APPEND VT_COMPILE_FLAGS ${CURFLAG})
       		list(REMOVE_AT asList 0)
		list(LENGTH asList length)
	endwhile()
#	message("asList: ${asList}")
#request all other flags
	execute_process(COMMAND ${VAMPIR_CXX} "-vt:show" "-vt:inst" "manual" RESULT_VARIABLE VT_RES OUTPUT_VARIABLE VT_OUT)
	string(REPLACE " " ";" asList ${VT_OUT} )
	list(LENGTH asList length)
#remove the compiler
	list(REMOVE_AT asList 0)
	set(VT_LIBRARIES )
	set(VT_LINKER_DIRECTORIES )
	set(VT_LINK_FLAGS "")
	while(length GREATER 1)
		list(GET asList 0 CURFLAG)
#if the current flag is not a compile flag, add it
		list(FIND VT_COMPILE_FLAGS ${CURFLAG} ISCOMPILEFLAG)
#		message("iscompileflag: ${ISCOMPILEFLAG}")
		if(ISCOMPILEFLAG EQUAL -1)
			string(SUBSTRING "${CURFLAG}" 0 2 FLAGBEG)
			if(${FLAGBEG} STREQUAL "-L")
				string(SUBSTRING "${CURFLAG}" 2 -1 CURLINKDIR)
				list(APPEND VT_LINKER_DIRECTORIES ${CURLINKDIR})
			elseif(${FLAGBEG} STREQUAL "-l")
				string(SUBSTRING "${CURFLAG}" 2 -1 CURLIBRARY)
				list(APPEND VT_LIBRARIES ${CURLIBRARY})
			endif()
			set(VT_LINK_FLAGS "${VT_LINK_FLAGS} ${CURFLAG}")
		endif()
       		list(REMOVE_AT asList 0)
		list(LENGTH asList length)
	endwhile()

	message("vt_compiler: ${VT_COMPILER}")
	message("vt_compiler flags: ${VT_COMPILE_FLAGS}")
	message("vt_linker flags: ${VT_LINK_FLAGS}")
	message("vt_linker directories: ${VT_LINKER_DIRECTORIES}")
	message("vt_linker libraries: ${VT_LIBRARIES}")
else(VAMPIR_CXX)
	set(VAMPIR_FOUND FALSE)
endif()
